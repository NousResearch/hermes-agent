# Relatório de Investigação — Bugs no fluxo OAuth da Anthropic (hermes-agent)

> **Estado atual do PR #87891:** este arquivo preserva os achados contra a implementação anterior. O fluxo de login OAuth do **dashboard web** (`_start_anthropic_pkce` / `_submit_anthropic_pkce` / `_save_anthropic_oauth_creds`) foi removido por completo, em vez de mantido com patches de CSRF ou de shadowing. Um endpoint HTTP não-supervisionado emitindo tokens de assinatura Claude Pro/Max fora do cliente oficial da Anthropic fica fora da política aceita para este produto. O catálogo marca `anthropic` como `flow: "external"`, e as rotas `start`/`submit` rejeitam esse fluxo. O fluxo PKCE interativo de terminal (`hermes auth add anthropic`) permanece explicitamente fora do escopo desta remoção.
>
> Os testes `tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py` e `tests/hermes_cli/test_web_server_oauth_write.py` foram removidos por testarem exclusivamente código inexistente na cabeça atual. Bugs 3 e 5 continuam cobertos: refresh Anthropic com lock cross-processo, lock adicional para o arquivo compartilhado `claude_code` (inclusive no resolver direto), write-through de `hermes_pkce`, e cobertura nativa Windows.

**Data do achado original:** 2026-08-16
**Branch do PR:** `fix/anthropic-oauth-csrf-race-apikey-shadow`
**Escopo investigado:** `agent/anthropic_adapter.py`, `agent/credential_pool.py`, `hermes_cli/auth.py`, `hermes_cli/auth_commands.py`, `hermes_cli/web_server.py`
**Cobertura atual:** `tests/hermes_cli/test_web_oauth_dispatch.py`, `tests/agent/test_credential_pool_anthropic_refresh_race.py`, `tests/agent/test_anthropic_oauth_stress.py`, `tests/agent/test_credential_pool_oauth_writethrough.py`, `tests/agent/test_anthropic_keychain.py`, `tests/hermes_cli/test_auth_store_lock_concurrent.py` e `web/src/lib/api.test.ts`

## Resumo executivo

Foram encontrados **dois bugs reais e reproduzíveis** na implementação anterior do fluxo OAuth da Anthropic, ambos confirmados por testes automatizados que falhavam contra aquela versão (evidência objetiva, não especulação):

| # | Bug | Categoria | Severidade |
|---|-----|-----------|------------|
| 1 | Dashboard web reintroduz vazamento do PKCE `code_verifier` via parâmetro `state` da URL de autorização | Segurança (CSRF / RFC 7636) | **Alta** |
| 2 | `_submit_anthropic_pkce()` do dashboard não valida o `state` retornado no callback — não há proteção CSRF nenhuma nesse endpoint | Segurança (CSRF / RFC 6749 §10.12) | **Alta** |
| 3 | Refresh de token OAuth da Anthropic não é protegido pelo lock cross-processo (`_auth_store_lock`) que Codex e xAI recebem, apesar de o refresh token da Anthropic também ser single-use | Race condition | **Média-Alta** |

Não foi encontrada evidência de **zombie processes** no fluxo Anthropic especificamente (ver seção 4) — essa hipótese do usuário não se confirmou para este provedor, ao contrário das duas hipóteses de segurança/race que se confirmaram.

---

## 1. Bug histórico de segurança — PKCE `code_verifier` vazado como `state` (dashboard web)

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

## 2. Bug histórico de segurança — ausência total de validação de `state` no callback do dashboard

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

`agent/credential_pool.py`, método `CredentialPool._refresh_entry()` (implementação anterior; as linhas atuais mudaram):

O trecho abaixo registra a condição **antes** do PR e não descreve o código atual:

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

O próprio comentário do código explica a razão de existir o lock: refresh tokens single-use não podem ser disputados por dois processos Hermes ao mesmo tempo. Na implementação anterior, **`"anthropic"` não estava na tupla `("openai-codex", "xai-oauth")`** — mesmo o refresh da Anthropic sendo, pela documentação do próprio arquivo `agent/anthropic_adapter.py::_refresh_oauth_token`, também single-use:

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

Ou seja: **só funcionava para credenciais que vieram do Claude Code CLI** (`~/.claude/.credentials.json`). Credenciais originadas do login OAuth nativo do próprio Hermes (`manual:hermes_pkce`, gravadas em `~/.hermes/.anthropic_oauth.json`, e também as antigas emitidas pelo dashboard — `manual:dashboard_pkce`) **não tinham nenhuma rota de recuperação**. Ao perder a corrida, o processo caía direto em `self._mark_exhausted(entry, None)`, mesmo que uma credencial válida já existisse em disco (escrita pelo processo vencedor).

### Cenário concreto de exploração / impacto

Isso não exige ataque — acontece em uso normal: múltiplos processos Hermes concorrentes (fleet workers, cron jobs, sessões CLI simultâneas) compartilhando a mesma credencial Anthropic via login PKCE nativo. Quando o token expira e dois processos tentam renovar simultaneamente:
1. Ambos leem o mesmo `refresh_token` (ainda válido) do disco.
2. Ambos disparam POST para `https://platform.claude.com/v1/oauth/token` quase ao mesmo tempo.
3. O servidor da Anthropic aceita apenas o primeiro (`refresh_token` é single-use); o segundo recebe `invalid_grant`.
4. O processo perdedor **não tem como se recuperar** e marca a credencial como `STATUS_EXHAUSTED` — falha espúria de "reautentique-se com a Anthropic" mesmo havendo um token perfeitamente válido em outro processo.

### Evidência de teste

Os testes abaixo são a evidência histórica da regressão na implementação anterior:

```
FAILED test_anthropic_refresh_is_not_protected_by_cross_process_lock
   assert []   # _auth_store_lock nunca foi chamado para 'anthropic'

FAILED test_concurrent_hermes_pkce_refresh_loses_credential_despite_valid_token_on_disk
   assert None is not None   # processo perdedor não conseguiu recuperar

PASSED test_concurrent_claude_code_refresh_recovers_via_credentials_file
   # contraste: fonte 'claude_code' SE recupera — prova a assimetria
```

O teste original usava um servidor OAuth fake que impunha corretamente a semântica "single-use" (primeiro a chegar ganha, segundo recebe `invalid_grant`) e disparava dois `CredentialPool._refresh_entry()` concorrentes via `threading.Thread`. A cobertura atual acrescenta processo independente, perfis distintos e contagem exata do POST; ver a seção 5.

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

## 5. Como validar a implementação atual

```bash
scripts/run_tests.sh tests/hermes_cli/test_web_oauth_dispatch.py tests/agent/test_credential_pool_anthropic_refresh_race.py tests/agent/test_anthropic_oauth_stress.py tests/agent/test_credential_pool_oauth_writethrough.py tests/hermes_cli/test_auth_store_lock_concurrent.py -q
```

O teste `test_distinct_profiles_share_one_claude_refresh_without_duplicate_post` usa dois processos independentes e exige um único POST de `stale-rt`; `test_hermes_pkce_refresh_writes_back_to_singleton` confirma que a rotação sobrevive a um `load_pool()` novo; `test_concurrent_refreshes_use_one_shared_credentials_lock` cobre o resolver direto; e os testes marcados `windows_only` exercitam a implementação `msvcrt` no host Windows. A cobertura frontend correspondente é `web/src/lib/api.test.ts`.

## 6. Arquivos e linhas de referência

| Arquivo | Linhas relevantes |
|---|---|
| `hermes_cli/web_server.py` | Catálogo `anthropic` como `flow: "external"`; dispatcher rejeita `start`/`submit` |
| `agent/credential_pool.py` | `_refresh_entry`, `_sync_anthropic_entry_from_pool_store`, lock compartilhado Claude Code e recuperação |
| `agent/anthropic_credentials.py` | `claude_code_credentials_path`, refresh puro, fluxo CLI PKCE separado e `CredentialPersistError` |
| `agent/anthropic_endpoints.py` | Predicados de família de endpoint por base URL |
| `agent/anthropic_message_convert.py` | Conversão de payload OpenAI → Anthropic |
| `agent/anthropic_adapter.py` | Construção de cliente e chamada da Messages API; re-exporta os três módulos acima |
| `web/src/lib/api.ts` | `/api/providers/oauth` incluído no escopo de perfil |

---

## 7. Validação manual pós-fix (2026-08-17)

Antes da remoção do fluxo dashboard, o shadowing de API key foi validado
manualmente com um teste A/B real — mesma cena, duas versões do código, sem
mocks. Este registro é histórico e não deve ser interpretado como prova de
que o endpoint dashboard ainda existe:

1. `.env` de teste com `ANTHROPIC_API_KEY` obsoleta (simulando um setup antigo
   esquecido) + chamada real a `_save_anthropic_oauth_creds` (a função que
   existia no login OAuth do dashboard), em `HERMES_HOME` isolado.
2. **Sem o fix** (`main` @ `8c8d55b`, app instalado do usuário): depois do
   login OAuth, o `.env` continuou com a key obsoleta e
   `resolve_anthropic_token()` seguiu retornando a API key
   (`is_oauth_token=False`) — bug reproduzido.
3. **Com o fix** (commit `41b7aba875`, esta branch): depois do mesmo login, o
   `.env` foi limpo automaticamente (`ANTHROPIC_API_KEY=`) e
   `resolve_anthropic_token()` passou a retornar o token OAuth
   (`sk-ant-oat...`, `is_oauth_token=True`).

O fluxo dashboard foi removido depois dessa validação. O que permanece
verificável localmente é o fluxo de terminal e a ausência das rotas dashboard;
uma validação em uma instalação publicada não foi executada nesta revisão e
depende de distribuir uma versão que contenha a cabeça final do PR.

---

## 8. Commit do refresh como parte da transação (2026-08-27)

O refresh token da Anthropic é single-use: o POST que devolve o par novo
invalida o que foi enviado. O par novo só existe de fato quando chega ao store
autoritativo — `~/.claude/.credentials.json` (`claude_code`) ou
`~/.hermes/.anthropic_oauth.json` (`hermes_pkce`). Esses arquivos são
autoritativos no sentido estrito: `credential_pool._seed_from_singletons()`
os relê em todo `load_pool()` e escreve o que encontra por cima da linha do
pool.

Até esta revisão os dois escritores engoliam `OSError`/`IOError` em nível
debug, então nenhum chamador distinguia commit durável de escrita perdida. Um
refresh podia gastar o único refresh token, reportar sucesso e deixar no disco
o par já consumido — que um restart re-semeava, fazendo o refresh seguinte
reenviar um token gasto.

O que mudou:

- `_write_claude_code_credentials()` / `_write_hermes_oauth_credentials()`
  levantam `CredentialPersistError` em vez de engolir o erro.
- `_refresh_oauth_token()` trata commit falho como refresh falho.
- `_refresh_entry_impl()` falha fechado nos caminhos primário e de retry: a
  rotação nunca é marcada, persistida ou devolvida, e a entrada vai para
  quarentena `DEAD` com razão `credential_persist_failed`, virando um pedido
  explícito de reautenticação em vez de um fallback silencioso para outro
  provedor. O retry commita no singleton antes de persistir a linha do pool.
- `_upsert_entry()` deixou de tratar a re-semeadura de uma fonte *borrowed*
  como rotação de chave (compara `secret_fingerprint`), senão cada load
  limpava o `DEAD` recém-escrito e ressuscitava a credencial em quarentena.

Cobertura: `tests/agent/test_anthropic_credential_persist_failure.py` força a
falha do escritor a partir de cada ponto de entrada e prova, em todos, que um
`load_pool()` posterior não devolve o par pré-refresh como credencial usável.

`agent/anthropic_adapter.py` foi dividido na mesma revisão (3.423 → 1.215
linhas) em `agent/anthropic_endpoints.py`, `agent/anthropic_message_convert.py`
e `agent/anthropic_credentials.py`, para que a superfície de credencial tenha
um único dono. O adapter re-exporta todos os nomes, então os imports existentes
continuam resolvendo.

