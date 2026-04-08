# Service Manager Normalization

Canonical runtime contract for Hermes services:

1. Linux: systemd is authoritative (`hermes-gateway.service` style units).
2. macOS: launchctl is authoritative (LaunchAgent plist under `~/Library/LaunchAgents`).
3. CLI wrappers (`hermes gateway start/stop`) must remain orchestration helpers, not service-state source of truth.

Guidelines:
- Keep one primary service definition per profile.
- Health/status commands should report both process status and last runtime status file.
- Restart policy: restart on failure for messaging adapters; clean stop for explicit shutdown.
- Do not mix multiple supervisors (e.g. launchctl + manual nohup) for the same profile.

Legacy wrapper policy:
- `scripts/hermes-gateway` is compatibility-only and must only forward to `hermes gateway ...`.
- No launchd/systemd unit should execute `scripts/hermes-gateway` directly.
- Canonical launchd ProgramArguments must run `python -m hermes_cli.main gateway run --replace`.

Verification checklist:
- `hermes gateway status` reports running state.
- PID file exists under the active profile home.
- Runtime status (`gateway.status.json`) has recent heartbeat/updated timestamp.
- `~/Library/LaunchAgents/ai.hermes.gateway*.plist` contains `hermes_cli.main` and not `scripts/hermes-gateway`.
- A restart reproduces adapter reconnection + cron ticking.
