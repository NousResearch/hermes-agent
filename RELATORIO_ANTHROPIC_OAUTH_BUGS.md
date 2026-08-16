# Relatório de Investigação — Bugs no fluxo OAuth da Anthropic (hermes-agent)

**Data:** 2026-08-16
**Branch:** `fix/update-orphan-history-guard-87694`
**Escopo investigado:** `agent/anthropic_adapter.py`, `agent/credential_pool.py`, `hermes_cli/auth.py`, `hermes_cli/auth_commands.py`, `hermes_cli/web_server.py`
**Testes novos:** `tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py`, `tests/agent/test_credential_pool_anthropic_refresh_race.py`

## Resumo executivo

Foram encontrados **dois bugs reais e reproduzíveis** no fluxo OAuth da Anthropic, ambos confirmados por testes automatizados que falham contra o código atual (evidência objetiva, não especulação):

| # | Bug | Categoria | Severidade |
|---|-----|-----------|------------|
| 1 | Dashboard web reintroduz vazamento do PKCE `code_verifier` via parâmetro `state` da URL de autorização | Segurança (CSRF / RFC 7636) | **Alta** |
| 2 | `_submit_anthropic_pkce()` do dashboard não valida o `state` retornado no callback — não há proteção CSRF nenhuma nesse endpoint | Segurança (CSRF / RFC 6749 §10.12) | **Alta** |
| 3 | Refresh de token OAuth da Anthropic não é protegido pelo lock cross-processo (`_auth_store_lock`) que Codex e xAI recebem, apesar de o refresh token da Anthropic também ser single-use | Race condition | **Média-Alta** |

Não foi encontrada evidência de **zombie processes** no fluxo Anthropic especificamente (ver seção 4) — essa hipótese do usuário não se confirmou para este provedor, ao contrário das duas hipóteses de segurança/race que se confirmaram.

---

## 1. Bug de segurança — PKCE `code_verifier` vazado como `state` (dashboard web)

### Onde

`hermes_cli/web_server.py`, função `_start_anthropic_pkce()` (~linha 10637-10661):

```python
def _start_anthropic_pkce(profile: Optional[str] = None) -> Dict[str, Any]:
    """Begin PKCE flow. Returns the auth URL the UI should open."""
    ...
    verifier, challenge = _generate_pkce_pair()
    sid, sess = _new_oauth_session("anthropic", "pkce", profile=profile)
    sess["verifier"] = verifier
    sess["state"] = verifier  # Anthropic round-trips verifier as state
    params = {
        ...
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": verifier,
    }
    auth_url = f"{_ANTHROPIC_OAUTH_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
```

### Por que é um bug real (não é estilo/preferência)

Este é **exatamente** o padrão vulnerável já documentado e corrigido no próprio repositório para o fluxo de CLI, em `agent/anthropic_adapter.py::run_hermes_oauth_login_pure()`. O teste de regressão `tests/agent/test_anthropic_oauth_pkce.py` explica o histórico:

> "Guards against re-introducing the bug where the PKCE `code_verifier` was reused as the OAuth `state` parameter, leaking the verifier via the authorization URL (browser history, Referer headers, auth-server logs) and removing CSRF protection on the callback path."
> — PR #1775 corrigiu; PR #2647 reintroduziu silenciosamente; PR #3107 removeu a função antiga; PR #10699 (issue #10693) corrigiu de novo na função sobrevivente.

O fluxo de CLI **foi corrigido** (gera `oauth_state = secrets.token_urlsafe(32)` independente do verifier). Mas o fluxo do **dashboard web** (`hermes_cli/web_server.py`) é uma implementação **paralela e independente** do mesmo login PKCE da Anthropic, e nunca recebeu a mesma correção — ele reintroduz o bug já resolvido, só que numa rota diferente (`/api/providers/oauth/anthropic/start` → `_start_anthropic_pkce`).

Consequência: o `code_verifier` (que por RFC 7636 §7.2 deve permanecer confidencial no servidor até a troca do código) vaza em:
- histórico do navegador do usuário;
- cabeçalho `Referer` de qualquer requisição subsequente feita a partir da página `claude.ai/oauth/authorize`;
- logs de acesso do servidor da Anthropic (`platform.claude.com`).

### Evidência de teste

`tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py` — 5 testes, todos falham contra o código atual:

```
FAILED test_authorization_url_state_is_not_pkce_verifier
FAILED test_verifier_never_appears_anywhere_in_auth_url
FAILED test_state_is_cryptographically_independent_of_verifier
FAILED test_submit_pkce_rejects_state_mismatch
FAILED test_submit_pkce_with_no_state_suffix_does_not_silently_succeed
```

Saída real do primeiro teste (state == verifier, comprovado byte a byte):

```
assert 'dg7ihLbmv6ooNu2-V_dKcVPKp29kNJewr4s8UBSr1gE' != 'dg7ihLbmv6ooNu2-V_dKcVPKp29kNJewr4s8UBSr1gE'
```

### Correção recomendada

Em `_start_anthropic_pkce()`, gerar um `state` independente (mesmo padrão de `agent/anthropic_adapter.py`):

```python
oauth_state = secrets.token_urlsafe(32)
sess["state"] = oauth_state
params = {..., "state": oauth_state}
```

---

## 2. Bug de segurança — ausência total de validação de `state` no callback do dashboard

### Onde

`hermes_cli/web_server.py`, função `_submit_anthropic_pkce()` (~linha 10664-10692):

```python
def _submit_anthropic_pkce(session_id, code_input, profile=None):
    ...
    parts = code_input.strip().split("#", 1)
    code = parts[0].strip()
    state_from_callback = parts[1] if len(parts) > 1 else ""

    exchange_data = json.dumps({
        "grant_type": "authorization_code",
        "client_id": _ANTHROPIC_OAUTH_CLIENT_ID,
        "code": code,
        "state": state_from_callback or sess["state"],   # <- nunca comparado
        "redirect_uri": _ANTHROPIC_OAUTH_REDIRECT_URI,
        "code_verifier": sess["verifier"],
    }).encode()
```

### Por que é um bug real

O código **nunca compara** `state_from_callback` com `sess["state"]`. Ele apenas usa o que veio do usuário (`state_from_callback`) ou, se vazio, cai de volta pro valor da própria sessão (`sess["state"]`) — ou seja, **qualquer valor** (ou nenhum) é aceito. Isso é diferente do fluxo de CLI (`agent/anthropic_adapter.py::run_hermes_oauth_login_pure`), que faz:

```python
if received_state != oauth_state:
    logger.warning("OAuth state mismatch — possible CSRF, aborting")
    return None
```

Sem essa checagem, o endpoint fica sem proteção CSRF conforme RFC 6749 §10.12 — é o segundo componente do mesmo problema descrito na Seção 1 (o vazamento do verifier como state e a ausência de validação se reforçam: mesmo que o `state` fosse aleatório, sem comparação a proteção seria inútil; e como é o verifier, mesmo com comparação a proteção colapsaria porque quem descobre o verifier vazado também "sabe" o state).

### Evidência de teste

```
FAILED test_submit_pkce_rejects_state_mismatch
FAILED test_submit_pkce_with_no_state_suffix_does_not_silently_succeed
```

O teste mostra a troca de token acontecendo mesmo com `state` adulterado (`attacker-controlled-state`) — o payload POST real foi capturado, provando que a troca prosseguiu sem checagem.

### Correção recomendada

```python
if not state_from_callback or state_from_callback != sess["state"]:
    with _oauth_sessions_lock:
        sess["status"] = "error"
        sess["error_message"] = "OAuth state mismatch"
    return {"ok": False, "status": "error", "message": "OAuth state mismatch"}
```

---

## 3. Race condition — refresh de OAuth da Anthropic sem lock cross-processo

### Onde

`agent/credential_pool.py`, método `CredentialPool._refresh_entry()` (~linha 1307-1344):

```python
# Codex and xAI OAuth refresh tokens are single-use. The
# sync→POST→write-back sequence below must run atomically across Hermes
# processes: otherwise two processes can both adopt the same on-disk
# token, both POST it, and the loser gets ``refresh_token_reused``.
# Serialize the whole sequence through the shared cross-process
# auth-store flock ...
if self.provider in ("openai-codex", "xai-oauth"):
    ...
    with _auth_store_lock(timeout_seconds=self._single_use_refresh_lock_timeout()):
        ...
return self._refresh_entry_impl(entry, force=force)   # <- anthropic cai aqui, SEM lock
```

O próprio comentário do código explica a razão de existir o lock: refresh tokens single-use não podem ser disputados por dois processos Hermes ao mesmo tempo. Só que **`"anthropic"` não está na tupla `("openai-codex", "xai-oauth")`** — mesmo o refresh da Anthropic sendo, pela documentação do próprio arquivo `agent/anthropic_adapter.py::_refresh_oauth_token`, também single-use:

> "Claude Code's OAuth refresh tokens are single-use: a successful refresh rotates the pair and invalidates the old refresh token."

### O único caminho de recuperação existe, mas é parcial

Ao falhar (`except Exception as exc`), há uma tentativa de recuperação em `_refresh_entry_impl` (linha ~1447):

```python
if self.provider == "anthropic" and entry.source == "claude_code":
    synced = self._sync_anthropic_entry_from_credentials_file(entry)
    ...
```

Só que `_sync_anthropic_entry_from_credentials_file` (linha 855-905) começa com:

```python
if self.provider != "anthropic" or entry.source != "claude_code":
    return entry
```

Ou seja: **só funciona para credenciais que vieram do Claude Code CLI** (`~/.claude/.credentials.json`). Credenciais originadas do login OAuth nativo do próprio Hermes (`hermes_pkce`, gravadas em `~/.hermes/.anthropic_oauth.json`, e também as emitidas pelo dashboard — `manual:dashboard_pkce`) **não têm nenhuma rota de recuperação**. Ao perder a corrida, o processo cai direto em `self._mark_exhausted(entry, None)` (linha 1705), mesmo que uma credencial válida já exista em disco (escrita pelo processo vencedor).

### Cenário concreto de exploração / impacto

Isso não exige ataque — acontece em uso normal: múltiplos processos Hermes concorrentes (fleet workers, cron jobs, sessões CLI simultâneas) compartilhando a mesma credencial Anthropic via login PKCE nativo. Quando o token expira e dois processos tentam renovar simultaneamente:
1. Ambos leem o mesmo `refresh_token` (ainda válido) do disco.
2. Ambos disparam POST para `https://platform.claude.com/v1/oauth/token` quase ao mesmo tempo.
3. O servidor da Anthropic aceita apenas o primeiro (`refresh_token` é single-use); o segundo recebe `invalid_grant`.
4. O processo perdedor **não tem como se recuperar** e marca a credencial como `STATUS_EXHAUSTED` — falha espúria de "reautentique-se com a Anthropic" mesmo havendo um token perfeitamente válido em outro processo.

### Evidência de teste

`tests/agent/test_credential_pool_anthropic_refresh_race.py` — 3 testes:

```
FAILED test_anthropic_refresh_is_not_protected_by_cross_process_lock
   assert []   # _auth_store_lock nunca foi chamado para 'anthropic'

FAILED test_concurrent_hermes_pkce_refresh_loses_credential_despite_valid_token_on_disk
   assert None is not None   # processo perdedor não conseguiu recuperar

PASSED test_concurrent_claude_code_refresh_recovers_via_credentials_file
   # contraste: fonte 'claude_code' SE recupera — prova a assimetria
```

O teste usa um servidor OAuth fake que impõe corretamente a semântica "single-use" (primeiro a chegar ganha, segundo recebe `invalid_grant`), exatamente como o comportamento real documentado da Anthropic, e dispara dois `CredentialPool._refresh_entry()` concorrentes via `threading.Thread` simulando dois processos Hermes distintos.

### Correção recomendada

Incluir `"anthropic"` na lista protegida por `_auth_store_lock` em `_refresh_entry()`, e generalizar `_sync_anthropic_entry_from_credentials_file` (ou adicionar uma variante) para também resincronizar a partir de `~/.hermes/auth.json` / `~/.hermes/.anthropic_oauth.json` quando `entry.source` for `hermes_pkce` ou `manual:dashboard_pkce`, análogo ao que já existe para `openai-codex` (`_sync_codex_entry_from_auth_store`) e `xai-oauth`.

---

## 4. Zombie processes — investigado, sem evidência no fluxo Anthropic

Verificado especificamente para OAuth da Anthropic:
- Não há servidor HTTP local nem `threading.Thread` de polling associado ao fluxo Anthropic (nem CLI nem dashboard) — diferente de outros provedores (Nous, Codex device-code) que usam thread de poll em background.
- `run_oauth_setup_token()` usa `subprocess.run([claude_path, "setup-token"])` — chamada **síncrona e bloqueante**, com `wait()` implícito; não deixa processo zumbi.
- `webbrowser.open(auth_url)` é chamado tanto no CLI quanto (implicitamente via browser do usuário) no dashboard — é comportamento padrão da stdlib do Python, idêntico em todos os outros provedores OAuth do repositório (xAI, Codex, Nous, Spotify, MiniMax), não é uma particularidade introduzida pelo fluxo Anthropic.

**Conclusão:** a hipótese de zombie process não se sustentou para este provedor especificamente. Caso o usuário tenha observado zumbis reais, provavelmente estão associados a outro subsistema (ex.: probes de processo do Windows já corrigidos em commits recentes desta branch — `4e3de140c1`, `00ecb5d538`) e não ao OAuth da Anthropic.

---

## 5. Como rodar os testes

```bash
python -m pytest tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py tests/agent/test_credential_pool_anthropic_refresh_race.py -v
```

Resultado atual (código não corrigido): **7 failed, 1 passed** — as 7 falhas são a evidência dos bugs 1, 2 e 3; o único teste que passa (`test_concurrent_claude_code_refresh_recovers_via_credentials_file`) prova a assimetria descrita no Bug 3 (fonte `claude_code` se recupera, `hermes_pkce`/dashboard não).

## 6. Arquivos e linhas de referência

| Arquivo | Linhas relevantes |
|---|---|
| `hermes_cli/web_server.py` | 10637-10661 (`_start_anthropic_pkce`), 10664-10728 (`_submit_anthropic_pkce`) |
| `agent/credential_pool.py` | 1307-1344 (`_refresh_entry`), 855-905 (`_sync_anthropic_entry_from_credentials_file`), 1442-1483 (fallback de recuperação), 1705 (`_mark_exhausted`) |
| `agent/anthropic_adapter.py` | 1125-1186 (`refresh_anthropic_oauth_pure`), 1189-1239 (`_refresh_oauth_token`, docstring sobre single-use), 1531-1658 (`run_hermes_oauth_login_pure`, fluxo CLI correto) |
| `tests/agent/test_anthropic_oauth_pkce.py` | Regressão histórica do bug 1/2 no fluxo CLI (já corrigido lá) |

---

## 7. Validação manual pós-fix (2026-08-17)

Além da suíte automatizada (seção 5), o fix do Bug 3 (API-key shadowing) foi
validado manualmente com um teste A/B real — mesma cena, duas versões do
código, sem mocks:

1. `.env` de teste com `ANTHROPIC_API_KEY` obsoleta (simulando um setup antigo
   esquecido) + chamada real a `_save_anthropic_oauth_creds` (a função
   disparada pelo login OAuth do dashboard), em `HERMES_HOME` isolado.
2. **Sem o fix** (`main` @ `8c8d55b`, app instalado do usuário): depois do
   login OAuth, o `.env` continuou com a key obsoleta e
   `resolve_anthropic_token()` seguiu retornando a API key
   (`is_oauth_token=False`) — bug reproduzido.
3. **Com o fix** (commit `41b7aba875`, esta branch): depois do mesmo login, o
   `.env` foi limpo automaticamente (`ANTHROPIC_API_KEY=`) e
   `resolve_anthropic_token()` passou a retornar o token OAuth
   (`sk-ant-oat...`, `is_oauth_token=True`).

Em seguida, o usuário rodou o `hermes chat` real no terminal (CLI, não
dashboard) com o binário instalado apontando para esta branch via
`PYTHONPATH`, contra seu `HERMES_HOME` real, e confirmou visualmente que o
app sobe e opera normalmente sob o código corrigido.

**Ação pendente:** o app instalado do usuário (`%LOCALAPPDATA%\hermes\hermes-agent`)
ainda está em `main` sem este commit — precisa dar merge no PR #87891 e
atualizar (`hermes update`) para o fix valer em produção, não só na branch de
teste.
