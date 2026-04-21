# Grok Provider Tests

Manual integration tests for the Patchright Grok provider.
Requires an active X Premium+ subscription and grok.com access.

## Prerequisites

```bash
pip install patchright && patchright install chromium
```

## First run

On first run, Chrome opens to grok.com. Log in with your X account.
Session saves to `~/.hermes/grok_profile/` and persists across runs.

## Test files

| File | What it tests | Browser? | Run |
|------|--------------|----------|-----|
| `test_prompt_formatter.py` | `_GrokPromptFormatter` edge cases (loop fix, error detection, sanitization, etc.) + `_MODE_LABELS` + `_set_mode` mocked unit tests | **NO** (pure-function, fast) | `python3 tests/grok/test_prompt_formatter.py` |
| `test_browser.py` | Browser launch, login detection, session persistence, editor found | YES | `python3 tests/grok/test_browser.py` |
| `test_adapter.py` | Full adapter interface, OpenAI-compatible response, multi-message | YES | `python3 tests/grok/test_adapter.py` |
| `test_response_parsing.py` | Short/long/code responses, edge cases | YES | `python3 tests/grok/test_response_parsing.py` |
| `test_mode_switcher.py` | `_set_mode()` actually clicks grok.com dropdown — switches through all 5 modes, verifies trigger label updates, idempotency, unknown mode safety | YES | `python3 tests/grok/test_mode_switcher.py` |
| `debug_mode_dropdown.py` | DOM inspection helper — opens grok.com, screenshots the mode dropdown, dumps selectors. Used to discover and verify `_MODE_LABELS` selectors. | YES | `python3 tests/grok/debug_mode_dropdown.py` |
| `debug_chat_persistence.py` | Thread-safe basic chat persistence probe (URL + page count per send). Superseded by `debug_chat_deep.py` for deep analysis. | YES | `python3 tests/grok/debug_chat_persistence.py` |
| `debug_chat_deep.py` | Deep DOM + network + URL diagnostic. Main-thread sync_api + event hooks (framenavigated, popup, request, response). Used to verify that raw Patchright flow maintains same chat. | YES | `python3 tests/grok/debug_chat_deep.py` |
| `debug_client_flow.py` | Uses real `PatchrightGrokClient` for 4 sends in sequence on the SAME client instance — confirms the adapter code maintains chat persistence when the client is reused. | YES | `python3 tests/grok/debug_client_flow.py` |
| `debug_client_lifecycle.py` | Verifies chat persistence across create/close cycles — mimics Hermes' per-request client lifecycle (create, send, close, repeat). All 4 sends must hit the same grok.com chat URL. | YES | `python3 tests/grok/debug_client_lifecycle.py` |
| `debug_auth_flow.py` | Interactive browser launcher for manual auth-flow capture / screenshots / debugging. Opens grok.com with the persistent profile and stays open until the user hits Enter (or SIGTERM if non-TTY). Not an automated test. | YES | `python3 tests/grok/debug_auth_flow.py` |

## Run order

1. `test_prompt_formatter.py` (pure-function, fast smoke test — run this first to catch logic bugs without spinning up a browser)
2. `test_browser.py` (verifies browser basics)
3. `test_adapter.py` (verifies adapter interface Hermes will use)
4. `test_response_parsing.py` (verifies different response types)

**IMPORTANT:** these are standalone scripts with their own PASS/FAIL counters, NOT pytest tests.
Do NOT run via `pytest tests/grok/` — that triggers xdist worker concurrency where multiple
browser contexts spawn against the same `~/.hermes/grok_profile/` profile dir, causing race
conditions and the appearance of orphaned tabs. Always invoke directly via `python3`.
