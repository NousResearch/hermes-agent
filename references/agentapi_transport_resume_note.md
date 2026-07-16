# Resume note: Antigravity `agentapi` transport

- Статус: реализован и проверен 2026-07-17.
- Коммиты: `4030c8d8a` (print-mode foundation) + `194c2143a` (agentapi transport).
- Stateful вызов: `delegate_task(..., acp_command="agy", acp_args=["agentapi"])`.
- Fallback: `delegate_task(..., acp_command="agy", acp_args=[])` → `agy -p`.
- Обязательный env: `ANTIGRAVITY_LS_ADDRESS`, `ANTIGRAVITY_PROJECT_ID`.
- Transport: `agent/antigravity_agentapi_client.py`.
- Shim/integration: `agent/copilot_acp_client.py`.
- Targeted tests: `24 passed`.
- Live direct two-turn smoke: `HERMES_AGENTAPI_LIVE_ONE` / `HERMES_AGENTAPI_LIVE_TWO`.
- Fresh Hermes delegation smoke: `AGY_AGENTAPI_DELEGATE_OK`.
- Полный отчет и команды: `references/agentapi_transport_handoff.md`.
- Опционально дальше: service auto-discovery и подготовка upstream PR.
