# Patches Locais — Hermes One vs Upstream

**Classificação:** Confirmada (git diff + leitura de código, 2026-08-10)  
**D-004 (2026-08-10):** model library **extraída** para `hermes_cli/hermes_one_model_library.py`; `web_server.py` só monta. OpenRouter prune permanece inline (candidato a PR upstream).

Estes patches existem no checkout local e **não estão no upstream** NousResearch. Risco em merges de `main` — reduzido após extração.

---

## 1. HERMES_ONE_MODEL_LIBRARY_COMPAT_V1

**Módulo:** `hermes_cli/hermes_one_model_library.py`  
**Mount:** `hermes_cli/web_server.py` (~3 linhas: import + `mount_hermes_one_model_library(app)`)

**Propósito:** Endpoints REST para biblioteca de modelos remotos — consumidor **Hermes One externo** ou SSH remote picker. O `apps/desktop` upstream usa `/api/model/options` e `/api/model/set` (**não** consome `/api/model/library`).

| Endpoint | Método | Função |
|---|---|---|
| `/api/model/library` | GET | Lista modelos + modelo ativo |
| `/api/model/library` | POST | Adiciona shortcut |
| `/api/model/library/{id}` | PATCH | Atualiza entrada |
| `/api/model/library/{id}` | DELETE | Remove entrada |

**Persistência:** `%LOCALAPPDATA%\hermes\models.json` (HERMES_HOME-aware via `get_hermes_home()`)

**Por que não `/api/plugins/...`:** clientes Hermes One hard-codam `/api/model/library`. Plugin dashboard monta sob `/api/plugins/<name>/` e quebraria o contrato. Extração modular + mount fino é o passo seguro; plugin completo exigiria hook de prefixo custom no core.

**Autenticação:** `/api/model/library` **não** está em `PUBLIC_API_PATHS` — requer session token.

---

## 2. OpenRouter credential pool prune

**Arquivo:** `agent/credential_pool.py`

**Propósito:** Quando `OPENROUTER_API_KEY` está ausente, remove entries do pool com `source=env:OPENROUTER_API_KEY` (evita zumbi — INCIDENTE-AUTH-JSON-REWRITE).

**Status D-004:** permanece patch local; candidato forte a **PR upstream** (bug fix genérico, sem Hermes One).

---

## 3. package-lock.json drift

Side-effect de `npm install` local — **não** é patch. Reverter se aparecer dirty.

---

## Estratégias D-004

| Etapa | Status |
|---|---|
| Manter patches manuais | ✅ baseline |
| Extrair model library para módulo + mount fino | ✅ feito 2026-08-10 |
| Plugin `/api/plugins/...` | ❌ bloqueado por contrato Hermes One |
| Hook core `register_api_mount(prefix=...)` | pendente (se quiser plugin puro) |
| PR upstream OpenRouter prune | pendente |

---

## Checklist pós-merge

- [ ] `python harness/scripts/hermes_patch_guard.py` → ok
- [ ] Grep `HERMES_ONE` em `hermes_one_model_library.py` + mount em `web_server.py`
- [ ] Grep `INCIDENTE-AUTH-JSON-REWRITE` em `credential_pool.py`
- [ ] Smoke GET/POST `/api/model/library` com `hermes serve` + token
