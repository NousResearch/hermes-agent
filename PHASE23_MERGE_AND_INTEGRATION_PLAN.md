# Hermes Stack Phase 23 -- Merge and Integration Plan

Date: 2026-07-02

## Current Merge-Ready Heads

| Repo | Branch | Head | Merge-readiness |
|---|---|---:|---|
| `hermes-agent` | `feat/runtime-run-api-contract` | `3bfde24` | Ready |
| `hermes-webui` | `feat/runtime-adapter-hermex-contract` | `573f40c` | Ready |

## Merge Evidence

Phase 22A completed the credential-aware merge-readiness gate.

Verified evidence:

- Pending-action smoke: PASSED
- Agent deterministic runtime smoke: PASSED, 7 passed and 0 failed
- Cross-repo deterministic smoke: PASSED, 11 passed and 0 failed
- Agent focused runtime tests: PASSED, 150 passed and 0 failed
- WebUI focused default tests: PASSED, 77 passed
- WebUI forced `agent-runs` env tests: EXPECTED PARTIAL, 69 passed and 8 known `tests/test_runtime_routes.py` failures
- DeepSeek real-provider smokes: SKIPPED because `DEEPSEEK_API_KEY` was absent
- Telegram live adapter smoke: SKIPPED because Telegram credentials and a safe private test chat were absent
- Cleanup: ports `8642` and `8789` free after Phase 22A

## Merge Order

Recommended order:

1. Merge `hermes-agent` first.
2. Merge `hermes-webui` second.
3. Run post-merge deterministic cross-repo smoke.
4. Integrate the merged runtime contract into Hermex.
5. Run Hermex provider-level smoke with configured fallback chain.

## Safe Merge Commands

Replace `<BASE_BRANCH>` with the actual base branch, most likely `main` unless this repo uses another target.

### hermes-agent

    cd ~/hermes-stack-work/hermes-agent
    git fetch origin
    git checkout <BASE_BRANCH>
    git pull --ff-only origin <BASE_BRANCH>
    git merge --no-ff feat/runtime-run-api-contract -m "Merge runtime run API contract"

### hermes-webui

    cd ~/hermes-stack-work/hermes-webui
    git fetch origin
    git checkout <BASE_BRANCH>
    git pull --ff-only origin <BASE_BRANCH>
    git merge --no-ff feat/runtime-adapter-hermex-contract -m "Merge Agent runs runtime adapter contract"

## Post-Merge Verification

Run after both merges:

    cd ~/hermes-stack-work/hermes-agent
    scripts/smoke_runtime_pending_actions.sh
    scripts/smoke_runtime_executor_live.sh --fake
    scripts/smoke_cross_repo.sh

    scripts/run_tests.sh \
      tests/gateway/test_runtime_run_manager.py \
      tests/gateway/test_runtime_routes.py \
      tests/gateway/test_runtime_approval_clarify.py \
      tests/gateway/test_runtime_executor.py \
      tests/gateway/test_runtime_executor_routes.py \
      tests/gateway/test_runtime_live_http_smoke.py

    cd ~/hermes-stack-work/hermes-webui
    ./scripts/test.sh \
      tests/test_agent_runs_adapter.py \
      tests/test_runtime_routes.py \
      tests/test_mobile_pending_actions.py \
      tests/test_agent_runs_live_http_smoke.py \
      -v

Optional forced `agent-runs` WebUI check:

    cd ~/hermes-stack-work/hermes-webui
    HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs \
    HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642 \
    HERMES_WEBUI_AGENT_RUNS_API_KEY=test-key \
    ./scripts/test.sh \
      tests/test_agent_runs_adapter.py \
      tests/test_runtime_routes.py \
      tests/test_mobile_pending_actions.py \
      tests/test_agent_runs_live_http_smoke.py \
      -v

Expected forced-mode result:

    69 passed, 8 known expected failures in tests/test_runtime_routes.py

## Hermes Integration

Hermes should consume the merged `hermes-agent` runtime API contract as the canonical execution surface.

Required environment:

    HERMES_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642
    HERMES_AGENT_RUNS_API_KEY=<local-or-deployed-runtime-key>

Contract:

- `POST /v1/runs`
- `GET /v1/runs/{run_id}`
- `GET /v1/runs/{run_id}/events`
- `POST /v1/runs/{run_id}/stop`
- `POST /v1/runs/{run_id}/approvals/{approval_id}`
- `POST /v1/runs/{run_id}/clarifications/{clarify_id}`

Required invariants:

- `RuntimeExecutor` owns `execute:true` runs.
- `RuntimeControlBridge` owns approval/clarify controls.
- No production-only pending-action injection endpoint exists.

## Hermes WebUI Integration

Required environment:

    HERMES_WEBUI_RUNTIME_ADAPTER=agent-runs
    HERMES_WEBUI_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642
    HERMES_WEBUI_AGENT_RUNS_API_KEY=<local-or-deployed-runtime-key>

Expected behavior:

- `/api/runtime/capabilities` reports `agent-runs`.
- `/api/runs/{run_id}` proxies Agent run status.
- `/api/runs/{run_id}/events` proxies Agent events.
- `/api/runs/{run_id}/cancel` maps to Agent stop.
- Approval and clarify resolution proxy through the Agent adapter.
- Mobile pending-actions surface approval/clarify actions from active runs.

## Hermex Integration

Hermex should treat `hermes-agent` as the runtime authority and use the WebUI adapter only as the UI/API presentation layer.

Recommended provider chain:

1. DeepSeek direct API as default fast provider.
2. OpenRouter PAYG as fallback provider.
3. Anthropic API for hard coding turns.
4. Gemini API as cheap/free backup.
5. OpenAI API with strict monthly cap for rescue/final review.

Hermex runtime binding:

    HERMEX_RUNTIME_PROVIDER=hermes-agent
    HERMEX_AGENT_RUNS_BASE_URL=http://127.0.0.1:8642
    HERMEX_AGENT_RUNS_API_KEY=<local-or-deployed-runtime-key>

If Hermex uses Hermes WebUI as its presentation layer:

    HERMEX_WEBUI_BASE_URL=http://127.0.0.1:8789
    HERMEX_WEBUI_RUNTIME_ADAPTER=agent-runs

## Post-Merge Release Gates

DeepSeek Agent-only smoke:

    cd ~/hermes-stack-work/hermes-agent
    DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" scripts/smoke_runtime_executor_live.sh

DeepSeek cross-repo smoke:

    cd ~/hermes-stack-work/hermes-agent
    DEEPSEEK_API_KEY="$DEEPSEEK_API_KEY" scripts/smoke_cross_repo.sh

Telegram adapter smoke requires:

    TELEGRAM_BOT_TOKEN=<token>
    TELEGRAM_CHAT_ID=<safe-private-test-chat>

Do not run Telegram smoke against production chats.

## Remaining Blockers

No deterministic merge blockers remain.

Credential-gated checks remain deferred:

- Real DeepSeek Agent-only smoke.
- Real DeepSeek cross-repo smoke.
- Telegram reference adapter smoke.

These are non-blocking unless release policy requires live-provider evidence before merge.

## Phase 23 Decision

Proceed to merge preparation.

Do not push to base branches until:

1. Base branches are confirmed.
2. Post-merge deterministic smokes pass.
3. No unexpected WebUI failures appear beyond the known forced `agent-runs` partial.
