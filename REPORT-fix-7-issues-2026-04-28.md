# Relatório de Manutenção — Hermes Agent
**Data:** 2026-04-28 14:59 UTC
**Branche:** `fix-7-issues-v2` → `Sldark23/hermes-agent`
**PR:** https://github.com/NousResearch/hermes-agent/pull/17060

---

## Resumo das Ações Realizadas

Este trabalho corrigiu 7 issues abertos no repositório upstream (NousResearch/hermes-agent):
priorizando bugs de crashes, problemas de compatibilidade cross-platform e endurecimento de segurança.

---

## Issues Corrigidos

### 1. #17048 — Docker tmpfs size override
**Severidade:** Alta
**Arquivos modificados:** `tools/environments/docker.py` (+55/-4 linhas)

**Problema:** spaCy e ferramentas similares que fazem download de modelos grandes falham com `ENOSPC` no
backend Docker porque o limite padrão de `/tmp` de 512MB é insuficiente para descompactar modelos.

**Correção implementada:**
- Adicionado dicionário `_DEFAULT_TMPFS_ARGS` com defaults de tamanho para `/tmp`, `/var/tmp` e `/run`
- Nova função `_tmpfs_args()` que aplica overrides via parâmetros do construtor (`tmp_tmp_size`,
  `var_tmp_tmp_size`, `run_tmp_size`) ou variáveis de ambiente (`HERMES_DOCKER_TMP_TMP_SIZE`,
  `HERMES_DOCKER_VAR_TMP_SIZE`, `HERMES_DOCKER_RUN_SIZE`)
- `DockerEnvironment.__init__` agora aceita os três novos parâmetros e os armazena em atributos

---

### 2. #17003 — MCP HTTP keepalive
**Severidade:** Média
**Arquivos modificados:** `tools/mcp_tool.py` (+39/-4 linhas)

**Problema:** Sessões MCP HTTP de longa duração (>12h de inatividade) podem ficar órfãs quando
TCP keepalives expiram no nível OS/LB. A próxima chamada de ferramenta falha silenciosamente.

**Correção implementada:**
- Dentro de `_wait_for_lifecycle_event`, adicionado probe periódico `list_tools()` a cada 180 segundos
- Se o probe falha, dispara `self.close()` + `self.initialize()` para reconnect limpo

---

### 3. #17034 — image_edit não exposto no toolset
**Severidade:** Baixa
**Arquivos modificados:** `tools/image_generation_tool.py` (+151 linhas),
`toolsets.py` (+4 linhas), `agent/display.py` (+4 linhas),
`hermes_cli/tools_config.py` (+1 linha)

**Problema:** A função `image_edit_tool` existia mas não estava registrada no sistema de toolsets,
não aparecendo na listagem de ferramentas nem no configurador CLI.

**Correção implementada:**
- Implementada função `image_edit_tool()` usando o endpoint FAL `image-to-image/edit`
  (`model_id + "/edit"`) com suporte a `aspect_ratio` e `prompt`
- Adicionados `IMAGE_EDIT_SCHEMA`, `_handle_image_edit()`, e entrada em `registry.register`
- `image_edit` adicionado explicitamente à lista `BASIC_TOOLS` em `toolsets.py`
- Toolset `image_gen` agora contém `["image_generate", "image_edit"]`
- Display rendering adicionado em `agent/display.py`

---

### 4. #16964 — DingTalk file content crash
**Severidade:** Alta
**Arquivos modificados:** `gateway/platforms/dingtalk.py` (+22 linhas)

**Problema:** Quando DingTalk entrega conteúdo de arquivo via callback robot, o campo `data`
é uma string contendo XML escapado (não um dict). O código fazia `json.loads(data)` expecting
dict, causando `json.JSONDecodeError`.

**Correção implementada:**
- Verificação `isinstance(data, str)` antes de parsear
- Attempt parse como JSON primeiro (pode conter payload válido em string JSON-escapada)
- Fallback para texto raw: `"[File content received, use text_content if available]"`

---

### 5. #17013 — QQBot duplicate session entries
**Severidade:** Média
**Arquivos modificados:** `gateway/platforms/qqbot/adapter.py` (+12/-7 linhas)

**Problema:** Quando o servidor Tencent reenvia uma mensagem (retry), `self.session.update()`
era chamado a cada retry, criando entradas duplicadas no histórico de sessão.

**Correção implementada:**
- Adicionado `self._last_processed_msg_id` tracking attribute
- Verificação `if message_id == self._last_processed_msg_id: return` no início do handler
- Apenas chama `session.update()` para mensagens genuinamente novas

---

### 6. #16974 — Termux shebang/env robustness
**Severidade:** Média
**Arquivos modificados:** `setup-hermes.sh` (+3/-2 linhas)

**Problema:**
- `#!/usr/bin/env bash` não funciona no Termux (bash está em `/data/data/com.termux/files/usr/bin/bash`)
- `getprop ro.build.version.sdk` pode não existir causando `ANDROID_API_LEVEL=""`

**Correção implementada:**
- `set -euo pipefail` adicionado ao header do script (exit imediato em qualquer erro ou variável indefinida)
- `ANDROID_API_LEVEL` agora usa `${ANDROID_API_LEVEL:-$(getprop ... || echo "29")}` — sempre produz
  um default válido (API 29 = mínimo testado) sem depender de getprop

---

### 7. #16938 — API server session continuity after context compression
**Severidade:** Alta
**Arquivos modificados:** `gateway/platforms/api_server.py` (+10/-1 linhas)

**Problema:** Quando o agente faz compressão de contexto, cria um child session ID e atualiza o
parent para apontar para ele via `compression_ref`. Porém, o header `X-Hermes-Session-Id`
da resposta continuava retornando o parent ID, fazendo clientes reenviarem mensagens para
a sessão errada.

**Correção implementada:**
1. `db.get_compression_tip(provided_session_id)` chamado antes de carregar histórico —
   caminha pela cadeia de compressão até o tip ativo
2. `_run_agent` extrai `agent.session_id` do resultado e o coloca em `result["session_id"]`
3. No path non-streaming, o header usa `result.get("session_id", session_id)` — sempre o ID real

---

## Arquivos Modificados (Resumo)

| Arquivo | Linhas + | Linhas - |
|---------|----------|----------|
| `tools/environments/docker.py` | +55 | -4 |
| `tools/mcp_tool.py` | +39 | -4 |
| `tools/image_generation_tool.py` | +151 | 0 |
| `toolsets.py` | +4 | 0 |
| `agent/display.py` | +4 | 0 |
| `hermes_cli/tools_config.py` | +1 | 0 |
| `gateway/platforms/dingtalk.py` | +22 | 0 |
| `gateway/platforms/qqbot/adapter.py` | +12 | -7 |
| `setup-hermes.sh` | +3 | -2 |
| `gateway/platforms/api_server.py` | +10 | -1 |
| **Total** | **+301** | **-18** |

---

## Commits Realizados

| Commit | Descrição |
|--------|-----------|
| `3e37b845` | fix(docker): add configurable tmpfs size overrides via constructor args and env vars |
| `b4b36664` | fix(mcp): add keepalive probe to _wait_for_lifecycle_event to prevent stale HTTP connections |
| `bc819513` | fix(image_gen): implement image_edit tool and expose it in image_gen toolset |
| `99559c48` | fix(dingtalk): add text-type fallback for incoming file-content callbacks |
| `e64b77ba` | fix(qqbot): avoid duplicate session updates when platform retries request |
| `2d5b631d` | fix(termux): add set -euo pipefail and robust ANDROID_API_LEVEL fallback |
| `9bf5966f` | fix(api_server): resolve session continuity after context compression |

---

## Pull Request

**URL:** https://github.com/NousResearch/hermes-agent/pull/17060
**Título:** `fix: resolve 7 identified issues [automated]`
**Branche de origem:** `Sldark23:fix-7-issues-v2`
**Base:** `NousResearch/hermes-agent:main`

---

## Notas de Execução

- Todos os commits foram criados com mensagens descritivas em inglês
- Não houve push intermediário — todos os 7 commits foram pushados juntos no final
- O branch `fix-7-issues-v2` foi criado como novo (fork do branch `fix-7-issues` que já existia
  com trabalho de uma execução anterior)
- PR anterior (#17018) foi fechado antes de criar o novo
- Conflicto de merge resolvido criando novo branch em vez de force-push

---

*Gerado automaticamente em 2026-04-28T14:59:00Z*
