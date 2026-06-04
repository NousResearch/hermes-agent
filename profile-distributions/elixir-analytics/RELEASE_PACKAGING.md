# Elixir Analytics Hermes Release Packaging

Use this guide before staging, committing, or opening a PR from the Hermes
worktree for the analytics agent.

## Package Lanes

### profile-distribution

Includes the user-facing Elixir analytics profile:

- `profile-distributions/elixir-analytics/config.yaml`
- `profile-distributions/elixir-analytics/SOUL.md`
- `profile-distributions/elixir-analytics/ROADMAP.md`
- `profile-distributions/elixir-analytics/skills/**`
- `profile-distributions/elixir-analytics/profile_plugins/**`

This package changes what the installed `elixir-analytics` profile knows and
which deterministic analytics tools it can call.

### hermes-runtime

Includes Hermes core changes required by the profile:

- profile-owned plugin discovery
- profile distribution installation support
- toolset exposure for `elixir-analytics-runner`
- tests that prove the profile distribution can be installed and the runner
  plugin is exposed

This package is the minimum Hermes-core surface needed for the analytics profile
to work without hardcoding analytics logic into generic gateway code.

### slack-gateway

Includes Slack transport health changes:

- Slack Socket Mode reconnect/recovery logging
- tests that prove the gateway logs a recovered connection after transport
  self-healing

This package can ship with runtime changes, but it should remain reviewable as
gateway reliability work rather than analytics prompt behavior.

## Checker

Run from the Hermes repo:

```bash
venv/bin/python scripts/check_elixir_analytics_release_packaging.py --strict
```

Use JSON output when preparing package-specific staging:

```bash
venv/bin/python scripts/check_elixir_analytics_release_packaging.py --json
```

`--strict` fails when changed files are unclassified. Local artifacts such as
`.hermes-bootstrap-complete` are reported separately and should not be staged.

## Release Rule

Do not stage the full Hermes dirty worktree as one commit.

Package in this order unless the user chooses otherwise:

1. `hermes-runtime`
2. `profile-distribution`
3. `slack-gateway`

After profile distribution changes, sync the installed profile before Slack E2E:

```bash
cp profile-distributions/elixir-analytics/skills/elixir-analytics/SKILL.md /Users/ritik/.hermes/profiles/elixir-analytics/skills/elixir-analytics/SKILL.md
cp profile-distributions/elixir-analytics/ROADMAP.md /Users/ritik/.hermes/profiles/elixir-analytics/ROADMAP.md
cp profile-distributions/elixir-analytics/profile_plugins/elixir-analytics-runner/__init__.py /Users/ritik/.hermes/profiles/elixir-analytics/profile_plugins/elixir-analytics-runner/__init__.py
```

Then restart the profile gateway and verify Slack Socket Mode reconnects.
