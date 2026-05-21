# mini-swe-agent (vendored, environments-only fork)

This is a stripped-down vendored snapshot of
[`SWE-agent/mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent),
used by hermes-lite **only for its execution-environment classes**
(`LocalEnvironment`, `DockerEnvironment`, `SingularityEnvironment`,
`SwerexModalEnvironment`).

Everything else has been removed:

- `models/`     — model adapters (used litellm; not needed)
- `agents/`     — agent classes (litellm-coupled)
- `run/`        — CLI runners
- `config/`     — config presets
- `tests/`, `docs/`, `mkdocs.yml` — unused

The remaining surface is the package `__init__`, the `Environment` protocol,
and the `environments/` + `utils/` subpackages. No litellm.

See the upstream repo for the original full project.
