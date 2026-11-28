# Version matrix

| Requirement | Minimum tested | Notes |
|---|---|---|
| Hermes gateway | `2026.9.24` (v0.23.1, commit `3b4215785882`) | Has `hermes webapp` (`--build-only`/`--status`/`--sign-out`) and `hermes_cli.webapp_access`. Verify, don't guess: `hermes webapp --help`. |
| Caddy | `2.6.x` (2.6.2 verified) | `caddy version`. |
| Node | any with `npm` | For the first renderer bundle build. |

Lock to a known-good commit of *this* skill after updating `hermes` on `main`.
