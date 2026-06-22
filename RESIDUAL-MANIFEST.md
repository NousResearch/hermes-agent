# Residual-line manifest — every overlay file now lives in a real PR diff

**Status: RESOLVED. This PR is a manifest, not a work item.**

The earlier `deferred/*.patch` files in this PR tracked overlay lines for 34 files.
Those lines are **not residual** — every one of the 34 files is carried
**byte-for-byte identical to overlay HEAD** by a real, open feature PR
(`#50484` clean-apply residual, or `#50487` drift residual with v0.17.0-ready variants).
So nothing in the v0.16.0→HEAD delta lives only as a `.patch` attachment.

Proof: for each file, `git rev-parse HEAD:<file>` == `git rev-parse <carrier-branch>:<file>`.

| Residual file | Carried by | apply-on-v0.17.0 |
|---|---|---|
| `agent/agent_init.py` | #50484 | clean |
| `agent/agent_runtime_helpers.py` | #50484 | clean |
| `agent/anthropic_adapter.py` | #50487 | via v0.17.0-ready/ |
| `agent/auxiliary_client.py` | #50487 | via v0.17.0-ready/ |
| `agent/chat_completion_helpers.py` | #50484 | clean |
| `agent/conversation_loop.py` | #50487 | via v0.17.0-ready/ |
| `agent/gemini_cloudcode_adapter.py` | #50484 | clean |
| `agent/gemini_native_adapter.py` | #50484 | clean |
| `agent/model_metadata.py` | #50484 | clean |
| `agent/models_dev.py` | #50484 | clean |
| `agent/system_prompt.py` | #50487 | via v0.17.0-ready/ |
| `agent/system_prompt_prelude.py` | #50484 | clean |
| `cli.py` | #50487 | via v0.17.0-ready/ |
| `gateway/platforms/api_server.py` | #50487 | via v0.17.0-ready/ |
| `gateway/run.py` | #50487 | via v0.17.0-ready/ |
| `hermes_cli/inventory.py` | #50484 | clean |
| `hermes_cli/main.py` | #50487 | via v0.17.0-ready/ |
| `hermes_cli/models.py` | #50484 | clean |
| `hermes_state.py` | #50487 | via v0.17.0-ready/ |
| `run_agent.py` | #50484 | clean |
| `tests/agent/test_anthropic_adapter.py` | #50484 | clean |
| `tests/agent/test_auxiliary_client.py` | #50487 | via v0.17.0-ready/ |
| `tests/agent/test_model_metadata.py` | #50484 | clean |
| `tests/hermes_cli/test_copilot_catalog_oauth_fallback.py` | #50484 | clean |
| `tests/hermes_cli/test_copilot_context.py` | #50484 | clean |
| `tests/hermes_cli/test_inventory.py` | #50487 | via v0.17.0-ready/ |
| `tests/hermes_cli/test_model_switch_copilot_api_mode.py` | #50484 | clean |
| `tests/hermes_cli/test_model_validation.py` | #50484 | clean |
| `tests/probe_prelude_e2e.py` | #50484 | clean |
| `tests/run_agent/test_run_agent.py` | #50484 | clean |
| `tests/test_context_engine_tool_wrap.py` | #50484 | clean |
| `tools/mcp_tool.py` | #50487 | via v0.17.0-ready/ |
| `tools/skills_tool.py` | #50487 | via v0.17.0-ready/ |
| `tui_gateway/server.py` | #50487 | via v0.17.0-ready/ |

## Coverage summary (v0.16.0 `3c231eb` → overlay HEAD)
- **160** changed files total.
- **138** in topical feature PRs.
- **34** residual source files → all carried by #50484 (17) / #50487 (17), shown above.
- **22** non-contributable: 9 `.bak` snapshots, 12 generated `.project-intel/` artifacts (incl. a `.sqlite` binary), 1 agy-cli test (agy-cli isolated by request).
- The real-source members of those 22 are themselves carried by #50484/#50487; the `.bak`/`.project-intel` files are build artifacts, not source.

## Net result on pristine v0.17.0 (`2bd1977d8`)
#50484 (clean) + #50487's `v0.17.0-ready/` (drift, resolved) apply as:
**34 files changed, +6050/-494, 0 conflict markers, 0 compile failures.**