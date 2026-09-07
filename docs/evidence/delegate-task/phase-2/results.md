Phase 2 target-profile runtime-scope result

Status: implementation evidence complete; audit pending
Base endpoint: 4d11a3057f4c9e85e918596f94e8b54ae52b2775
Implementation scope: target-profile runtime scope only

Changed production files
- agent/profile_runtime_scope.py: neutral context-local home, secret and terminal scope.
- gateway/run.py: compatibility wrappers delegate to neutral scope; async hydration remains off event loop.
- tools/delegate_tool.py: profile home resolution, top-level profile schema, profile defaults/toolsets, explicit pin provenance, construction scope, child execution scope.
- tools/delegate_tool_config.py: explicit profile credential isolation for API key/base URL.

Tests added/updated
- tests/tools/test_delegate_profile.py
- tests/tools/test_delegate_profile_runtime_scope.py
- Phase 1 provider/completion tests are carried forward unchanged and remain deferred RED.

Verification
- Owned Phase 2: 14 passed, 1 deselected (fallback deferred).
- Existing gateway/profile scope regressions: 15 passed, 1 warning.
- Collection: 29 tests collected, exit 0.
- Deferred Phase 3/5 matrix: 10 assertion-level failures, 13 passes, exit 1.
- AST parse and import checks: pass; import emitted existing missing toolaria/blobstore warning.
- Ruff: unavailable; no replacement lint tool installed.

Boundaries
- No provider-routing overlay or fallback implementation was added.
- No completion-unit/aggregator implementation was added.
- No merge, push, gateway restart, activation or deployment occurred.
