# QA Report: status realtime dos cards de agentes Ágora

## Objetivo
Validar que o dashboard Ágora reflete o estado real dos agentes com base nos workers ativos do Kanban, e que o backend de roteamento de @mentions também considera worker ativo.

## Mudanças aplicadas (t_a8b9aa1d)

### Frontend (`plugins/agora/dashboard/dist/index.js`)
- `enrichedAgents` agora sobrescreve `state` para `"working"` quando existe `workerMap[a.profile]`.
- Metadados do card (task_id, run_id, pid, heartbeat) vêm do worker ativo quando disponível.
- Agente sem worker ativo continua exibindo o estado de `agora_agent_status` (pode ser stale).

### Backend (`plugins/agora/dashboard/plugin_api.py`)
- Importa `hermes_cli.kanban_db` com fallback seguro.
- `_kanban_active_profiles()` consulta `task_runs` + `tasks` para obter perfis com worker ativo.
- `_deliver_mentions_to_tmux()` força `state="working"` para recipients ativos no Kanban quando o estado em `agora_agent_status` é `idle`/vazio, fazendo com que @mentions sejam entregues como `/steer` em vez de `prompt`.

## Validação no dashboard real

- Abas abertas: 1 (reutilizada via CDP).
- URL: `http://127.0.0.1:9119/agora`
- `/api/plugins/kanban/workers/active`: count=1 (run do próprio agora-frontend em t_a8b9aa1d).
- `/api/plugins/agora/agents/status`: 5 agentes conhecidos.
- DOM:
  - `agora-frontend`: `working` com task `t_a8b9aa1d` (worker ativo).
  - `agora-qa`: `reviewing` sem worker ativo (estado de agora_agent_status).
  - `agora-backend`: `idle`.

## Testes

```bash
python -m pytest tests/plugins/test_agora_dashboard_plugin.py -q
# 91 passed

python -m pytest tests/scripts/test_agora_notify.py -q
# 9 passed

node --check plugins/agora/dashboard/dist/index.js
# OK
```

## Artefatos

- `workers_active.json`
- `agents_status.json`
- `agent_cards_dom.json`
- `dashboard_agents.png`

## Conclusão

Frontend e backend agora dão prioridade ao worker ativo do Kanban para status e roteamento de mentions. Agentes sem worker ativo continuam a mostrar o último estado relatado em `agora_agent_status`.
