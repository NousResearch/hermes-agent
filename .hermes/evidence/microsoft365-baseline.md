# Microsoft 365 plugin baseline

Captured from commit `14f66491ad288c216e5e8a20eb57cc3097e36118` on branch `feat/microsoft365-connector`.

## Environment

- Command: `git rev-parse HEAD`
  - Result: `14f66491ad288c216e5e8a20eb57cc3097e36118`
- Command: `git status --short`
  - Result: pre-existing untracked `MICROSOFT365_PLUGIN_REVIEW_BRIEF.md`
- Command: `py -3.11 --version`
  - Result: `Python 3.11.9`
- Command: `py -3.11 -c 'import importlib.metadata as m; ...'`
  - Result: `msgraph-sdk=1.62.0`, `azure-identity=1.25.3`

No credentials, tokens, tenant data, configuration values, or `.env` contents were read or written.

## Verification commands

- `py -3.11 -m pytest tests/plugins/test_microsoft365_plugin.py -q`
  - Exit `0`; `16 passed in 2.17s`
- `py -3.11 -m pytest tests/tools/test_microsoft_graph_client.py tests/tools/test_microsoft_graph_auth.py -q`
  - Exit `1`; `16 passed, 1 failed, 8 warnings in 1.56s`
  - Failure: `TestMicrosoftGraphTokenProvider.test_concurrent_calls_share_one_token_fetch[trio]` because an `asyncio.gather` future was passed to Trio (`TypeError: trio.run received unrecognized yield message`).
  - This baseline evidence does not classify the failure as pre-existing without a clean-origin comparison.
- `py -3.11 -m hermes_cli.plugin_validate plugins/microsoft365/plugin.yaml`
  - Exit `0`; no output
- `py -3.11 -m compileall -q plugins/microsoft365`
  - Exit `0`
- `git diff --check origin/main...HEAD`
  - Exit `0`

## Scope guard

This evidence file is the only baseline artifact. No production code, `config/`, or `.env` file was modified during capture.
