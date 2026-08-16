# Anthropic OAuth — Issues encontrados no hermes-agent

> **Atualização:** todos os 5 bugs abaixo (incluindo o Bug 5, achado pelo teste de esforço) foram **corrigidos** nesta sessão.
> Issues: [#87887](https://github.com/NousResearch/hermes-agent/issues/87887) (bugs 1-2) · [#87888](https://github.com/NousResearch/hermes-agent/issues/87888) (bugs 3-4) · [#87889](https://github.com/NousResearch/hermes-agent/issues/87889) (bug 5).

**Data:** 2026-08-16
**Branch:** `fix/update-orphan-history-guard-87694`
**Pedido original:** investigar se o OAuth da Anthropic tem bug real — rotas, race conditions, zombie processes, segurança — e provar com testes.

---

## TL;DR — o login do Claude Code funciona?

**Sim, funciona.** Nenhum dos bugs encontrados impede login ou uso normal. São dois problemas distintos, ambos reais e comprovados por teste, mas nenhum é "quebrado/não autentica":

| Fluxo de login | Afetado por quê |
|---|---|
| `claude setup-token` (CLI oficial Anthropic, credencial lida de `~/.claude/.credentials.json`, `source="claude_code"`) | Nenhum dos bugs 1-3 — é o único caminho protegido nos dois casos |
| `hermes model` no terminal, opção 1 (OAuth via CLI) | Bug de race condition (só sob múltiplos processos Hermes concorrentes); limpa `ANTHROPIC_API_KEY` corretamente, então **não** sofre o Bug 4 |
| Login via **dashboard web** do Hermes | Bug de segurança (CSRF/PKCE leak, bugs 1-2) + race condition (bug 3) + **Bug 4: não limpa `ANTHROPIC_API_KEY` antiga, que continua vencendo o OAuth** |

**Atualização:** o sintoma relatado pelo usuário ("configuro com OAuth e o Hermes usa API key mesmo assim") foi diagnosticado como o **Bug 4** — ver seção dedicada abaixo.

- **Bug de segurança:** não impede login, mas expõe o fluxo do dashboard a um vetor de CSRF já corrigido no CLI e nunca replicado lá.
- **Bug de race condition:** só aparece com **múltiplos processos Hermes rodando ao mesmo tempo** (fleet workers, cron + sessão interativa) disputando o refresh do mesmo token no exato instante em que ele expira. Uso single-process (maioria dos usuários, a maior parte do tempo) nunca bate nisso.

---

## O que foi feito nesta investigação

1. Localizado o código real do fluxo Anthropic (não documentação): `agent/anthropic_adapter.py`, `agent/credential_pool.py`, `hermes_cli/auth.py`, `hermes_cli/auth_commands.py`, `hermes_cli/web_server.py`.
2. Comparado o fluxo de login via CLI (`run_hermes_oauth_login_pure`) com o fluxo paralelo do dashboard web (`_start_anthropic_pkce` / `_submit_anthropic_pkce`) — achado o bug de CSRF por essa comparação.
3. Comparado a proteção cross-processo que Codex/xAI recebem no refresh (`_auth_store_lock`) com o que a Anthropic recebe — achado o bug de race condition por essa comparação.
4. Escritos dois arquivos de teste novos que **provam** os bugs (testes que devem falhar contra o código atual, e falham):
   - `tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py` (5 testes)
   - `tests/agent/test_credential_pool_anthropic_refresh_race.py` (3 testes)
5. Rodado `pytest` nos dois arquivos: **7 failed, 1 passed** — confirma os bugs objetivamente (não é opinião).
6. Investigado zombie process especificamente no fluxo Anthropic — **não encontrado**.
7. Nenhum código de produção foi alterado — só testes de regressão + este relatório.

---

## Bug 1 — PKCE `code_verifier` vazado como `state` (só no dashboard web)

**Onde:** `hermes_cli/web_server.py`, função `_start_anthropic_pkce()` (~linha 10637-10661)

```python
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
```

Isso é **exatamente** o bug que já existiu no fluxo de CLI e foi corrigido (histórico documentado em `tests/agent/test_anthropic_oauth_pkce.py`: PR #1775 corrigiu → PR #2647 reintroduziu → PR #3107 removeu a função antiga → PR #10699/issue #10693 corrigiu de vez na função sobrevivente). O CLI (`agent/anthropic_adapter.py::run_hermes_oauth_login_pure`) gera um `state` independente (`secrets.token_urlsafe(32)`). O **dashboard web nunca recebeu essa correção** — é uma implementação paralela que reintroduz o problema numa rota diferente.

**Consequência:** o `code_verifier` (que por RFC 7636 §7.2 deve ficar confidencial) vaza via histórico do navegador, cabeçalho `Referer`, e logs de acesso da Anthropic (`platform.claude.com`).

**Evidência (teste real rodado):**
```
assert 'dg7ihLbmv6ooNu2-V_dKcVPKp29kNJewr4s8UBSr1gE' != 'dg7ihLbmv6ooNu2-V_dKcVPKp29kNJewr4s8UBSr1gE'
```
state e verifier são literalmente o mesmo valor.

**Status:** ✅ **corrigido** — `_start_anthropic_pkce()` agora gera `state` independente via `secrets.token_urlsafe(32)`.

---

## Bug 2 — Callback do dashboard nunca valida o `state` (zero proteção CSRF)

**Onde:** `hermes_cli/web_server.py`, função `_submit_anthropic_pkce()` (~linha 10664-10692)

```python
state_from_callback = parts[1] if len(parts) > 1 else ""
exchange_data = json.dumps({
    ...
    "state": state_from_callback or sess["state"],   # nunca comparado com nada
    ...
}).encode()
```

O código nunca faz `if state_from_callback != sess["state"]: reject`. Qualquer valor (ou nenhum) é aceito e a troca de token prossegue. O CLI faz essa checagem (`if received_state != oauth_state: abort`); o dashboard não faz nenhuma.

**Evidência (teste real rodado):** payload POST real capturado mesmo com `state` adulterado (`attacker-controlled-state`) — a troca de token aconteceu do mesmo jeito.

**Status:** ✅ **corrigido** — `_submit_anthropic_pkce()` agora rejeita a troca de token quando o `state` não bate.

---

## Bug 3 — Refresh de token sem lock cross-processo (race condition)

**Onde:** `agent/credential_pool.py`, `CredentialPool._refresh_entry()` (~linha 1307-1344)

```python
# Codex and xAI OAuth refresh tokens are single-use. [...]
# Serialize the whole sequence through the shared cross-process auth-store flock
if self.provider in ("openai-codex", "xai-oauth"):
    with _auth_store_lock(...):
        ...
return self._refresh_entry_impl(entry, force=force)   # anthropic cai aqui, SEM lock
```

`"anthropic"` não está na lista protegida, apesar de o refresh token da Anthropic também ser single-use (documentado no próprio `agent/anthropic_adapter.py::_refresh_oauth_token`: "a successful refresh rotates the pair and invalidates the old refresh token").

O único caminho de recuperação em caso de falha (`_sync_anthropic_entry_from_credentials_file`, linha 855-905) só funciona se `entry.source == "claude_code"`:
```python
if self.provider != "anthropic" or entry.source != "claude_code":
    return entry
```
Credenciais de login nativo do Hermes (`hermes_pkce`) ou do dashboard (`manual:dashboard_pkce`) **não têm recuperação nenhuma**. Se perdem a corrida, caem em `self._mark_exhausted(entry, None)` (linha 1705) — mesmo com um token válido existindo em disco, escrito pelo processo que ganhou.

**Cenário real (não ataque, uso normal):** dois processos Hermes concorrentes (ex: fleet worker + sessão CLI, ou dois cron jobs) compartilhando a mesma conta Anthropic via login nativo. Token expira; ambos tentam refresh ao mesmo tempo; servidor da Anthropic aceita só o primeiro; o segundo recebe `invalid_grant` e fica marcado como exhausted — pedindo reautenticação mesmo com credencial válida disponível.

**Evidência (teste real rodado, com servidor OAuth fake simulando single-use de verdade):**
```
FAILED test_anthropic_refresh_is_not_protected_by_cross_process_lock
   assert []   # _auth_store_lock nunca foi chamado para 'anthropic'

FAILED test_concurrent_hermes_pkce_refresh_loses_credential_despite_valid_token_on_disk
   assert None is not None   # processo perdedor não conseguiu recuperar

PASSED test_concurrent_claude_code_refresh_recovers_via_credentials_file
   # contraste: fonte 'claude_code' SE recupera — prova a assimetria
```

**Status:** ✅ **corrigido**. `"anthropic"` foi incluído na tupla protegida por `_auth_store_lock`, e um novo método `_sync_anthropic_entry_from_pool_store()` (espelhando o padrão já usado pelo xAI) resincroniza a partir do próprio credential-pool store — funciona pra **todas** as fontes (`claude_code`, `hermes_pkce`, `manual:dashboard_pkce`), não só `claude_code`.

---

## Bug 4 — API key antiga vence sobre OAuth (causa do "não funciona" relatado)

**Sintoma relatado pelo usuário:** faz o setup com OAuth da Anthropic e seleciona os modelos, mas o Hermes acaba usando modo API key (pay-per-token) em vez do plano OAuth (Claude Pro/Max).

**Onde:** `agent/anthropic_adapter.py::resolve_anthropic_token()` (~linha 1387-1442)

Ordem de prioridade na resolução de credencial:

```
1. ANTHROPIC_TOKEN               (OAuth salvo pelo Hermes via CLI)
2. CLAUDE_CODE_OAUTH_TOKEN
3. ANTHROPIC_API_KEY              <- API key explícita, sempre vence, por design
4. ~/.claude/.credentials.json   (Claude Code CLI)
5. credential_pool OAuth entry   (~/.hermes/auth.json) <- onde o login do dashboard fica
```

O comentário do próprio código é explícito sobre a intenção: "An explicit user-configured key must not be shadowed by auto-discovered [...] credential-pool OAuth credentials." — ou seja, se `ANTHROPIC_API_KEY` estiver preenchida (mesmo que antiga/esquecida), ela sempre ganha da credencial OAuth do pool (prioridade 3 bate prioridade 5), mesmo depois de um login OAuth bem-sucedido.

**A causa raiz específica:** o fluxo de login OAuth do **dashboard web** (`hermes_cli/web_server.py::_save_anthropic_oauth_creds`) grava a credencial no credential pool, mas **nunca limpa** a variável `ANTHROPIC_API_KEY`. Isso é diferente do fluxo de OAuth via **CLI** (`hermes model` → opção 1 → `save_anthropic_oauth_token()`), que limpa a API key antiga automaticamente ao salvar o token OAuth. Resultado: se em qualquer momento anterior uma `ANTHROPIC_API_KEY` ficou configurada (setup antigo, teste, auto-detecção), fazer login OAuth pelo dashboard não a remove, e ela continua vencendo pra sempre na resolução de credencial.

**Correção recomendada:** em `_save_anthropic_oauth_creds()` (web_server.py), limpar `ANTHROPIC_API_KEY` do mesmo jeito que `save_anthropic_oauth_token()` já faz no fluxo de CLI — ou, na resolução (`resolve_anthropic_token`), preferir a credencial OAuth mais recente quando o usuário acabou de completar um login explícito, em vez de uma API key estática que pode estar obsoleta.

**Diagnóstico que o usuário deve rodar (não incluído aqui por conter dados sensíveis — chaves de API não devem aparecer neste relatório):**
```bash
hermes doctor
# verificar se ANTHROPIC_API_KEY está preenchida em ~/.hermes/.env
```
Se estiver preenchida, é essa a causa. Correção rápida: esvaziar `ANTHROPIC_API_KEY` no `.env` do Hermes, ou refazer o login OAuth pela opção 1 do `hermes model` no terminal (que já limpa corretamente).

**Status:** ✅ **corrigido**. `_save_anthropic_oauth_creds()` agora limpa `ANTHROPIC_API_KEY` ao salvar o login OAuth do dashboard, igual o fluxo de CLI já fazia.

---

## Bug 5 — PermissionError sob concorrência real no lock cross-processo (achado pelo teste de esforço)

**Como foi achado:** ao escrever um teste de carga (`tests/agent/test_anthropic_oauth_stress.py`) simulando 20 "processos Hermes" concorrentes disputando o refresh (necessário pra validar o Bug 3 sob estresse, não só com 2 threads), o teste **falhou de verdade** — 16 de 20 threads levantaram `PermissionError: [Errno 13] Permission denied` vindo de dentro do próprio `_auth_store_lock()`.

**Causa raiz:** `hermes_cli/auth.py::_file_lock()` — a checagem "garante que o arquivo de lock tem pelo menos 1 byte" (necessária pro `msvcrt.locking()` do Windows) fazia um `write_text()` **sem tratamento de exceção**, fora do loop de retry que existe logo depois:

```python
if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
    lock_path.write_text(" ", encoding="utf-8")   # <- sem try/except
```

Sob concorrência real no Windows, essa escrita pode colidir com o byte-range lock (`msvcrt.locking`) que outra thread/processo já segura no mesmo arquivo nesse exato instante, levantando `PermissionError` — só que isso acontece **antes** do loop `while True: try: ... except (BlockingIOError, OSError, PermissionError): retry` que trataria exatamente esse tipo de colisão. Resultado: a exceção escapa sem ser tratada.

**Importante:** isso não é específico da Anthropic. `_file_lock()`/`_auth_store_lock()` é o mesmo primitivo compartilhado usado por Codex, xAI e Nous — qualquer provedor com refresh token single-use passando por esse lock estava exposto, só nunca tinha sido testado sob essa carga de concorrência antes.

**Correção aplicada:**
```python
if msvcrt and (not lock_path.exists() or lock_path.stat().st_size == 0):
    try:
        lock_path.write_text(" ", encoding="utf-8")
    except (OSError, PermissionError):
        pass  # outro holder já garantiu conteúdo; segue pro loop de retry
```

**Status:** ✅ **corrigido**. Testes: `tests/agent/test_anthropic_oauth_stress.py` (reproduz de forma confiável, 16/20 falhas antes do fix, 0 depois) e `tests/hermes_cli/test_auth_store_lock_concurrent.py` (cobertura dedicada e genérica do lock, independente da Anthropic).

---

## Zombie process — investigado, não confirmado

Checado especificamente no fluxo Anthropic:
- Sem servidor HTTP local nem thread de polling (diferente de Nous/Codex device-code).
- `run_oauth_setup_token()` usa `subprocess.run(...)` — bloqueante, com wait implícito, sem zumbi.
- `webbrowser.open(auth_url)` é comportamento padrão da stdlib, idêntico em todos os outros provedores do repo — não é peculiaridade da Anthropic.

**Conclusão:** hipótese de zombie process não se confirmou para este provedor.

---

## Arquivos criados nesta investigação

| Arquivo | O que é |
|---|---|
| `tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py` | 5 testes provando bugs 1 e 2 |
| `tests/agent/test_credential_pool_anthropic_refresh_race.py` | 3 testes provando bug 3 |
| `RELATORIO_ANTHROPIC_OAUTH_BUGS.md` | Relatório técnico detalhado (versão anterior desta investigação, bugs 1-3) |
| `ANTHROPIC_ISSUES.md` | Este arquivo — resumo consolidado, inclui bug 4 (causa real do "não funciona" relatado) |

**Nota sobre dados sensíveis:** este relatório foi verificado com busca por regex de chaves reais (`sk-ant-...`, `ANTHROPIC_API_KEY=`, `ANTHROPIC_TOKEN=`) — nenhuma encontrada. Só há valores sintéticos usados nos testes automatizados (ex: `sk-ant-oat-rotated-1`), nunca uma chave real. O diagnóstico do Bug 4 pede pro usuário rodar `hermes doctor` / checar o próprio `.env` localmente, mas o resultado desse comando **não foi colado neste relatório** — só a explicação da causa raiz no código.

Nenhum arquivo de produção foi alterado.

## Como reproduzir

```bash
python -m pytest tests/hermes_cli/test_anthropic_dashboard_pkce_csrf.py tests/agent/test_credential_pool_anthropic_refresh_race.py -v
```

Resultado com o código atual (não corrigido): **7 failed, 1 passed**.

## Referência rápida de linhas

| Arquivo | Linhas |
|---|---|
| `hermes_cli/web_server.py` | 10637-10661 (`_start_anthropic_pkce`), 10664-10728 (`_submit_anthropic_pkce`) |
| `agent/credential_pool.py` | 1307-1344 (`_refresh_entry`), 855-905 (`_sync_anthropic_entry_from_credentials_file`), 1442-1483 (recuperação), 1705 (`_mark_exhausted`) |
| `agent/anthropic_adapter.py` | 1125-1186 (`refresh_anthropic_oauth_pure`), 1189-1239 (`_refresh_oauth_token`), 1531-1658 (`run_hermes_oauth_login_pure`, fluxo CLI correto) |
