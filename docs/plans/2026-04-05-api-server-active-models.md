# API Server Active Models Plan

> For Hermes: implement with strict TDD. Write the failing targeted tests first, run them to confirm failure, then make the minimum code changes to pass.

Goal: Make the OpenAI-compatible Hermes API expose real active/configured models through GET /v1/models instead of only the synthetic hermes-agent alias, and let request-time model selection flow through chat/responses requests.

Architecture:
- Keep backward compatibility by retaining the hermes-agent alias.
- Add API-server helpers that derive the current runtime route plus authenticated/configured provider model entries.
- Use the existing shared model-switch pipeline so per-request model IDs resolve the same way as the existing /model command.

Files:
- Modify: gateway/platforms/api_server.py
- Modify: tests/gateway/test_api_server.py

Phase 1 tasks:
1. Add failing endpoint tests for /v1/models returning real configured models alongside hermes-agent.
2. Add failing request-routing tests proving chat/responses forward the requested model into agent creation/runtime selection.
3. Implement API-server model listing helper using existing gateway/runtime config plus hermes_cli.model_switch.list_authenticated_providers.
4. Implement request-time model override resolution using hermes_cli.model_switch.switch_model.
5. Run targeted tests, then focused regression tests for gateway API server behavior.

Verification:
- python3 -m pytest tests/gateway/test_api_server.py -q -k "models or requested_model"
- python3 -m pytest tests/gateway/test_api_server.py -q
