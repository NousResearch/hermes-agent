# Hosted artifacts end-to-end tests

Real `hermes chat -q` turns against a temp `HERMES_HOME` signed in to Nous, driven by the repo's
scripted fake LLM provider (`tests/fakes/fake_llm_provider.py`). Hermes' own MCP client talks to the
hosted artifacts MCP server (`hermes-artifacts`, Streamable HTTP at `<base_url>/v1/artifacts/mcp`) with
the profile's Nous access token as the bearer.

| File | What it proves |
|---|---|
| `test_artifact_edit_flow.py` | create → update from a stale base (the model sees a `stale_base` tool error with `head_version`) → update from that head → show. On the fake: the server received exactly those `tools/call`s with those arguments, every request carried the Nous bearer, the entitlement came from a fresh `/api/oauth/account` read with no token refresh. `content_url` is never printed to stdout. |
| `test_artifacts_fail_closed.py` | Account says `artifacts_access` false or absent, `paid_access` false, `artifacts.enabled: false`, or the Portal is down → no `hermes-artifacts` tool is offered to the model (every provider request's `tools[]`, including the Tool Search catalog) and the server receives nothing. An `entitled` control case proves the harness does attach. |

Fakes: `tests/fakes/hosted_artifacts_server.py` (real `mcp` SDK server, subprocess) and
`tests/fakes/nous_portal_account.py` (in-process Portal: `GET /api/oauth/account` with a configurable
body; a refresh `POST /api/oauth/token` is answered 400 and recorded).

Run them with the repo runner:

```bash
scripts/run_tests.sh tests/e2e/core/hosted_artifacts/test_artifact_edit_flow.py
scripts/run_tests.sh tests/e2e/core/hosted_artifacts/test_artifacts_fail_closed.py
```

## Pointing the edit flow at a real artifacts plane

Every test in `test_artifact_edit_flow.py` runs as `[fake]` and `[real]`. `[real]` is skipped unless
all of these are set:

| Variable | Value |
|---|---|
| `HERMES_ARTIFACTS_E2E_BASE_URL` | The plane's origin, e.g. `https://artifacts.example.test` (Hermes appends `/v1/artifacts/mcp`). Plain `http` is accepted only for `localhost`, `127.0.0.1`, `::1` and `*.localhost` hosts. |
| `HERMES_ARTIFACTS_E2E_PORTAL_URL` | The Portal origin the plane trusts; Hermes reads `<portal>/api/oauth/account` from it and must get `artifacts_access: true` and `paid_service_access.paid_access: true`. |
| `HERMES_ARTIFACTS_E2E_TOKEN_FILE` | Path to a file holding a Nous Portal access token for that user (JWT). Hermes sends it as the bearer to both the Portal and the plane. Give it at least 10 minutes of remaining life: the test login has no usable refresh token. |
| `HERMES_ARTIFACTS_E2E_TOKEN` | The token itself; an alternative to `_TOKEN_FILE` for a direct `pytest` run only. `scripts/run_tests.sh` forwards no credential-valued variable, so use `_TOKEN_FILE` with the runner. |

On `[real]`, the assertions that need the fake server's inbound log (exact wire calls, bearer on
every request, Portal reads) skip; the model-side assertions (stale error with `head_version`, retry
from the head, show, no `content_url` on stdout) run. The scenario creates one artifact titled
`Release notes` on the plane and does not delete it.

Command:

```bash
HERMES_ARTIFACTS_E2E_BASE_URL=... HERMES_ARTIFACTS_E2E_PORTAL_URL=... HERMES_ARTIFACTS_E2E_TOKEN_FILE=... \
  scripts/run_tests.sh tests/e2e/core/hosted_artifacts/test_artifact_edit_flow.py
```

A pass shows the `[real]` cells as passed, not skipped. The runner starts tests from an empty
environment and forwards only the three variables above (its `TEST_ENV` allowlist).

## Probe: does this Hermes support hosted artifacts?

```bash
python -c "import tools.hosted_artifacts as m; print(m.HOSTED_ARTIFACTS_CLIENT_VERSION)"
```

Run from the repository root (or with it on `PYTHONPATH`). It exits 0 and prints an integer `>= 1`
when this client can attach the hosted artifacts server; an `ImportError`/non-zero exit means it
cannot. The number is bumped only when the client's side of the wire contract changes.
