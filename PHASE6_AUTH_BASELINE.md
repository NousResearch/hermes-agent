# Phase 6 — Bootstrap dependency inventory

Source: clean One Gateway baseline at `bc1b572e2c`, copied to an independent
Phase 6 worktree. Counts are broad text matches, NOT confirmed import edges.

## Initial tracked-file census
Production grep targeted `hermes_cli.auth`, `hermes_cli.credential_lifecycle`,
`hermes_cli.secrets_cli`, and `agent.credential_pool`, `.credential_sources`,
`.credential_persistence`, `.secret_sources` in major runtime packages.
142 matching source files: agent 32; gateway 8; hermes_cli 87;
plugins 10; providers 1; tui_gateway 4.
A narrower credential reference grep matched 236 test files.
Classify each reference before moving it: comments and presentation-only
paths are included in this broad baseline; not all files require migration.

## Confirmed source seams to inspect
- `hermes_cli/auth.py`: mixed provider registry, auth store, locking and state.
- `hermes_cli/auth_*`: mixed interactive commands, OAuth/runtime refresh.
- `hermes_cli/auth_plugin_providers.py`: plugin metadata mirror and hooks.
- `hermes_cli/credential_lifecycle.py`: coordinated .env/pool/config cleanup.
- `agent/credential_pool.py`, `_admin.py`, `_model_cooldowns.py`, `_plugin.py`:
  runtime pooling and refresh; depend on CLI-owned auth.
- `agent/credential_sources.py`, `agent/credential_persistence.py`:
  credential-specific provenance, suppression, and borrowed state.
- `agent/secret_sources/`: currently generic; move only credential-specific
  responsibilities if a distinct auth owner is justified.
- `providers/base.py`: existing auth_handler/refresh_credential plugin contract.
- Gateway and TUI consumers include `gateway/session_policy_credentials.py`,
  `gateway/run_turn_runner.py`, `tui_gateway/agent_factory.py`.

## Regression seams
- tests/agent/test_credential_pool_provider_boundary.py
- tests/agent/test_credential_pool_profile_oauth_fork.py
- tests/agent/test_credential_pool_nous_refresh_stampede.py
- tests/agent/test_credential_pool_deferred_refresh.py
- tests/agent/test_multiplex_cloud_credential_clients.py
- tests/hermes_cli/test_auth_store_windows_encoding.py
- tests/hermes_cli/test_auth_toctou_file_modes.py
- tests/hermes_cli/test_credential_lifecycle.py
- tests/providers/test_auth_registry_import_order.py
- tests/providers/test_auth_registry_mid_discovery.py

## Constraints
- No `auth/` or `nous_cli/` package existed on the baseline.
- Existing Arcana query fails: malformed repository manifest (wrong field
  count). Do not block Phase 6 or attempt full rescan just to bootstrap.
- Tests must use isolated homes and mocked token endpoints, not live secrets.

Baseline targeted pytest: `python -m pytest -q -x` with
`tests/auth/test_boundary.py`, `tests/agent/test_credential_pool_provider_boundary.py`,
`tests/agent/test_credential_pool_nous_refresh_stampede.py`, and
`tests/hermes_cli/test_auth_store_windows_encoding.py`:
**21 passed in 15.64s** on Windows (2026-10-01).

Packaging check: `setuptools.find_packages(include=['auth', 'auth.*'])`
found `auth`. `tests/auth/test_boundary.py` plus
`tests/test_packaging_metadata.py`: **4 passed in 2.42s**.
`pyproject.toml` explicitly includes `auth` and `auth.*` in package discovery.
The pre-existing root `/auth/` gitignore pattern protects generated private
state; bootstrap adds a narrow Python-only exception for the source package.
Any future auth subpackages require matching deliberate source exceptions.
