# Anthropic OAuth — Issues encontrados no hermes-agent

> **Estado atual do PR #87891:** os achados abaixo são históricos e devem ser lidos junto com o estado atual. O fluxo OAuth Anthropic do dashboard foi removido, não corrigido em linha: o catálogo expõe `flow: "external"` e as rotas `start`/`submit` rejeitam o fluxo. O fluxo interativo de terminal (`hermes auth add anthropic`) permanece explicitamente fora do escopo desta remoção.
> Issues: [#87887](https://github.com/NousResearch/hermes-agent/issues/87887) (bugs 1-2) · [#87888](https://github.com/NousResearch/hermes-agent/issues/87888) (bugs 3-4) · [#87889](https://github.com/NousResearch/hermes-agent/issues/87889) (bug 5).
>
> **Decisão de segurança:** um endpoint HTTP não-supervisionado emitindo tokens de assinatura Claude Pro/Max fora do cliente oficial da Anthropic fica fora da política aceita para este produto. A correção é eliminar essa superfície, não mantê-la com patches de CSRF ou de persistência.
>
> **Controles que permanecem:** a serialização de refresh inclui `anthropic`; a fonte compartilhada `claude_code` usa um lock baseado no arquivo compartilhado tanto no `CredentialPool` quanto no resolver direto; o write-through de `hermes_pkce` atualiza `.anthropic_oauth.json`; e o lock Windows trata a inicialização concorrente do arquivo. A cobertura atual inclui teste com processos independentes e perfis distintos.

**Data:** 2026-08-16
**Branch do PR:** `fix/anthropic-oauth-csrf-race-apikey-shadow`
**Pedido original:** investigar se o OAuth da Anthropic tem bug real — rotas, race conditions, zombie processes, segurança — e provar com testes.

---

## TL;DR — o login do Claude Code funciona?

**Sim, funciona.** Nenhum dos bugs encontrados impede login ou uso normal. São dois problemas distintos, ambos reais e comprovados por teste, mas nenhum é "quebrado/não autentica":

| Fluxo de login | Estado no PR #87891 |
|---|---|
| `claude setup-token` (CLI oficial Anthropic, credencial lida de `~/.claude/.credentials.json`, `source="claude_code"`) | Permanece disponível; o refresh compartilhado é serializado e revalidado |
| `hermes auth add anthropic` no terminal | Permanece disponível e escreve a credencial Hermes; o refresh de `hermes_pkce` tem write-through |
| Login Anthropic no **dashboard web** do Hermes | Removido; o catálogo retorna `flow: "external"` e não há emissão de tokens por HTTP |

**Nota sobre o sintoma original:** o caminho dashboard que podia deixar uma API key antiga sombrear o OAuth não existe mais. A prioridade de uma API key explícita na resolução continua sendo deliberada para credenciais configuradas pelo usuário.

- **Bug de segurança histórico:** não impede login, mas expunha o fluxo do dashboard a um vetor de CSRF. A superfície HTTP foi removida no PR; o fluxo de terminal continua separado.
- **Bug de race condition:** só aparece com **múltiplos processos Hermes rodando ao mesmo tempo** (fleet workers, cron + sessão interativa) disputando o refresh do mesmo token no exato instante em que ele expira. Uso single-process (maioria dos usuários, a maior parte do tempo) nunca bate nisso.

---

## O que foi feito nesta investigação

1. Localizado o código real nos módulos `agent/anthropic_adapter.py`, `agent/credential_pool.py`, `hermes_cli/auth.py`, `hermes_cli/auth_commands.py` e `hermes_cli/web_server.py`.
2. Comparado o fluxo interativo de terminal com a implementação paralela do dashboard e confirmado que os bugs de CSRF/PKCE pertenciam à superfície HTTP removida.
3. Comparado o lock cross-processo de Codex/xAI com a Anthropic e alinhado a autoridade do lock à fonte real: `auth.json` por perfil e `~/.claude/.credentials.json` para `claude_code`.
4. Mantida a regressão de refresh em `tests/agent/test_credential_pool_anthropic_refresh_race.py`, ampliada a carga em `tests/agent/test_anthropic_oauth_stress.py` e coberto o lock Windows em `tests/hermes_cli/test_auth_store_lock_concurrent.py`.
5. Adicionada cobertura de dispatcher para provar que o dashboard não cria sessão nem emite URL/código OAuth Anthropic.
6. Investigado zombie process especificamente no fluxo Anthropic — **não encontrado**.
7. A documentação desta investigação foi atualizada para refletir a remoção do fluxo dashboard e os testes atuais; não representa uma reprodução contra o código atual quando descreve o estado anterior.

---

## Bug 1 (histórico) — PKCE `code_verifier` vazado como `state` (só no dashboard web)

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

Isso era **exatamente** o bug que já existiu no fluxo de CLI e foi corrigido (histórico documentado em `tests/agent/test_anthropic_oauth_pkce.py`: PR #1775 corrigiu → PR #2647 reintroduziu → PR #3107 removeu a função antiga → PR #10699/issue #10693 corrigiu de vez na função sobrevivente). O CLI (`agent/anthropic_adapter.py::run_hermes_oauth_login_pure`) gera um `state` independente (`secrets.token_urlsafe(32)`). O dashboard web tinha uma implementação paralela; no PR #87891 essa implementação foi removida, portanto o código vulnerável abaixo não é mais um call path ativo.

**Consequência:** o `code_verifier` (que por RFC 7636 §7.2 deve ficar confidencial) vaza via histórico do navegador, cabeçalho `Referer`, e logs de acesso da Anthropic (`platform.claude.com`).

**Evidência (teste real rodado):**
```
assert 'dg7ihLbmv6ooNu2-V_dKcVPKp29kNJewr4s8UBSr1gE' != 'dg7ihLbmv6ooNu2-V_dKcVPKp29kNJewr4s8UBSr1gE'
```
state e verifier são literalmente o mesmo valor.

**Status no PR:** ✅ **neutralizado por remoção** — `_start_anthropic_pkce()` não existe mais e o catálogo Anthropic é `flow: "external"`. O teste atual verifica que a rota `start` rejeita o fluxo, em vez de testar uma função removida.

---

## Bug 2 (histórico) — Callback do dashboard nunca valida o `state` (zero proteção CSRF)

**Onde:** `hermes_cli/web_server.py`, função `_submit_anthropic_pkce()` (~linha 10664-10692)

```python
state_from_callback = parts[1] if len(parts) > 1 else ""
exchange_data = json.dumps({
    ...
    "state": state_from_callback or sess["state"],   # nunca comparado com nada
    ...
}).encode()
```

O código antigo nunca fazia `if state_from_callback != sess["state"]: reject`. Qualquer valor (ou nenhum) era aceito e a troca de token prosseguia. O CLI faz essa checagem (`if received_state != oauth_state: abort`); o dashboard não tem mais esse endpoint.

**Evidência (teste real rodado):** payload POST real capturado mesmo com `state` adulterado (`attacker-controlled-state`) — a troca de token aconteceu do mesmo jeito.

**Status no PR:** ✅ **neutralizado por remoção** — `_submit_anthropic_pkce()` não existe mais; a rota `submit` genérica rejeita o provider Anthropic e não cria nem completa uma sessão OAuth.

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
Credenciais de login nativo do Hermes (`manual:hermes_pkce`, gravadas em `~/.hermes/.anthropic_oauth.json`, e a forma sem prefixo usada pelo seeding) ou as antigas emitidas pelo dashboard (`manual:dashboard_pkce`) **não tinham recuperação nenhuma**. Ao perder a corrida, o processo caía em `self._mark_exhausted(entry, None)` — mesmo com um token válido existindo em disco, escrito pelo processo vencedor.

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

**Status no PR:** ✅ **corrigido para os fluxos que permanecem**. `"anthropic"` foi incluído na tupla protegida por `_auth_store_lock`; `claude_code` também usa um lock keyed ao arquivo compartilhado `~/.claude/.credentials.json`; e `_sync_anthropic_entry_from_pool_store()` cobre fontes persistidas do pool. O dashboard `manual:dashboard_pkce` foi removido, não é mais uma fonte emitida.

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

**A causa raiz específica:** o fluxo de login OAuth do **dashboard web** (`hermes_cli/web_server.py::_save_anthropic_oauth_creds`) gravava a credencial no credential pool, mas **não limpava** a variável `ANTHROPIC_API_KEY`. Esse fluxo foi removido no PR. O fluxo de OAuth via **CLI** (`hermes model` → opção 1 → `save_anthropic_oauth_token()`), que limpa a API key antiga automaticamente ao salvar o token OAuth, é separado e permanece fora do escopo.

**Correção recomendada:** em `_save_anthropic_oauth_creds()` (web_server.py), limpar `ANTHROPIC_API_KEY` do mesmo jeito que `save_anthropic_oauth_token()` já faz no fluxo de CLI — ou, na resolução (`resolve_anthropic_token`), preferir a credencial OAuth mais recente quando o usuário acabou de completar um login explícito, em vez de uma API key estática que pode estar obsoleta.

**Diagnóstico que o usuário deve rodar (não incluído aqui por conter dados sensíveis — chaves de API não devem aparecer neste relatório):**
```bash
hermes doctor
# verificar se ANTHROPIC_API_KEY está preenchida em ~/.hermes/.env
```
Se estiver preenchida, é essa a causa. Correção rápida: esvaziar `ANTHROPIC_API_KEY` no `.env` do Hermes, ou refazer o login OAuth pela opção 1 do `hermes model` no terminal (que já limpa corretamente).

**Status no PR:** ✅ **neutralizado por remoção** — não existe mais `_save_anthropic_oauth_creds()` nem login Anthropic no dashboard. A prioridade de `ANTHROPIC_API_KEY` na resolução continua deliberada para uma chave explicitamente configurada.

---

## Bug 5 — PermissionError sob concorrência real no lock cross-processo (achado pelo teste de esforço)

**Como foi achado:** ao escrever um teste de carga (`tests/agent/test_anthropic_oauth_stress.py`) simulando concorrência real no refresh (necessário pra validar o Bug 3 sob estresse, não só com uma chamada isolada), a inicialização do lock no Windows podia levantar `PermissionError: [Errno 13] Permission denied` vindo de dentro do próprio `_auth_store_lock()`.

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

**Status no PR:** ✅ **corrigido**. O write de inicialização do lock é best-effort e qualquer contenção segue para o loop de retry. A cobertura do lock é `windows_only`, e o teste de refresh com perfis distintos usa processos independentes e exige exatamente um POST do refresh token compartilhado.

---

## Zombie process — investigado, não confirmado

Checado especificamente no fluxo Anthropic:
- Sem servidor HTTP local nem thread de polling (diferente de Nous/Codex device-code).
- `run_oauth_setup_token()` usa `subprocess.run(...)` — bloqueante, com wait implícito, sem zumbi.
- `webbrowser.open(auth_url)` é comportamento padrão da stdlib, idêntico em todos os outros provedores do repo — não é peculiaridade da Anthropic.

**Conclusão:** hipótese de zombie process não se confirmou para este provedor.

---

## Arquivos e cobertura atuais do PR

| Arquivo | O que é |
|---|---|
| `tests/hermes_cli/test_web_oauth_dispatch.py` | Dispatcher: Anthropic external; `start`/`submit` dashboard rejeitados |
| `tests/agent/test_credential_pool_anthropic_refresh_race.py` | 3 testes de lock/sincronização Anthropic |
| `tests/agent/test_anthropic_oauth_stress.py` | Carga em threads + processo cross-profile com POST único |
| `tests/agent/test_anthropic_keychain.py` | Resolver direto Claude Code sob lock compartilhado |
| `tests/hermes_cli/test_auth_store_lock_concurrent.py` | Concorrência do lock real, marcada `windows_only` |
| `tests/agent/test_credential_pool_oauth_writethrough.py` | Rotação `hermes_pkce` preservada no `.anthropic_oauth.json` |
| `RELATORIO_ANTHROPIC_OAUTH_BUGS.md` | Relatório técnico histórico e estado da implementação |
| `ANTHROPIC_ISSUES.md` | Este arquivo — resumo consolidado e matriz de escopo |

**Nota sobre dados sensíveis:** este relatório foi verificado com busca por regex de chaves reais (`sk-ant-...`, `ANTHROPIC_API_KEY=`, `ANTHROPIC_TOKEN=`) — nenhuma encontrada. Só há valores sintéticos usados nos testes automatizados (ex: `sk-ant-oat-rotated-1`), nunca uma chave real. O diagnóstico do Bug 4 pede pro usuário rodar `hermes doctor` / checar o próprio `.env` localmente, mas o resultado desse comando **não foi colado neste relatório** — só a explicação da causa raiz no código.

Arquivos de produção alterados pelo PR: `agent/anthropic_adapter.py`, `agent/credential_pool.py`, `hermes_cli/auth.py` e `hermes_cli/web_server.py`. O frontend também recebe a correção de escopo em `web/src/lib/api.ts` para que operações OAuth sigam o perfil selecionado.

## Como validar o estado atual

```bash
scripts/run_tests.sh tests/hermes_cli/test_web_oauth_dispatch.py tests/agent/test_credential_pool_anthropic_refresh_race.py tests/agent/test_anthropic_oauth_stress.py tests/agent/test_credential_pool_oauth_writethrough.py tests/agent/test_anthropic_keychain.py tests/hermes_cli/test_auth_store_lock_concurrent.py -q
```

Para a cobertura frontend de perfil OAuth:

```bash
cd web
npx vitest run src/lib/api.test.ts
```

Os resultados dependem do ambiente e devem ser registrados no PR junto com a cabeça testada; os blocos históricos acima não são resultados do código atual.

## Referência rápida de linhas

| Arquivo | Linhas |
|---|---|
| `hermes_cli/web_server.py` | Catálogo `anthropic` como `flow: "external"`; dispatcher rejeita `start`/`submit` |
| `agent/credential_pool.py` | `_refresh_entry`, `_sync_anthropic_entry_from_pool_store`, lock compartilhado Claude Code e fallback de recuperação |
| `agent/anthropic_adapter.py` | `claude_code_credentials_path`, refresh puro, fluxo CLI PKCE separado e write-through Hermes |
| `web/src/lib/api.ts` | `/api/providers/oauth` incluído no escopo de perfil do dashboard |
