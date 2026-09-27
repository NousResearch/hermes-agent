# Matter host image

This directory builds and boots a litco-agent matter host: one DigitalOcean droplet per real matter, created from a versioned snapshot. The snapshot carries code and packages but no secrets. Each droplet gets its secrets and its matter identity from cloud-init user-data at first boot.

The design follows `MATTER_AGENT_VM_PILOT_2026-09-27.md` section 5 in the litkit-notes repo, and the turn server it runs is described in `docs/litco.md`.

## Files

| File | Role |
|---|---|
| `install-host.sh` | The install steps, run as root on the machine being built. The droplet build and the smoke container both run it. |
| `build-image.sh` | Runs on the operator's machine. It creates a throwaway builder droplet, runs `install-host.sh` there, cleans cloud-init state, snapshots `litco-agent-host-<version>`, and deletes the builder. `--dry-run` prints the plan and touches nothing. |
| `cloud-init.yaml.tmpl` | Per-matter user-data template. |
| `render-user-data.py` | Renders the template from the control plane's JSON spec. Standard library only. |
| `litco-agent.service` | The systemd unit: the Hermes gateway with the `litco_turn` platform, run as `hermes`. |
| `litco-agent-init` | `ExecStartPre`. It renders `$HERMES_HOME/config.yaml` and `SOUL.md` from `profile/` using non-secret variables only. |
| `litco-agent-drain` | `ExecStop`. It waits until the turn server reports no running turn. |
| `profile/config.yaml.tmpl`, `profile/SOUL.md` | The matter profile templates. |
| `smoke.sh`, `smoke/` | Local container smoke test. |
| `Makefile` | `make -C deploy/host test`, `dry-run`, `smoke`. |

## What the image contains

| Path | Contents |
|---|---|
| `/opt/litco-agent/app` | The litco-agent checkout at the pinned ref, owned by root. |
| `/opt/litco-agent/app/.venv` | The Hermes venv, built with `uv sync --frozen --extra messaging` from `uv.lock`. |
| `/opt/litco-agent/python` | uv-managed CPython 3.14. Ubuntu 24.04's own SQLite (3.45) has the WAL-reset bug, and the uv build bundles SQLite 3.53. The install script refuses a Python whose SQLite is older than 3.51.3. |
| `/opt/litco-agent/tools` | The PM store: `agent-browser` and the PM-pinned Chromium, found by Hermes through `HERMES_RUNTIME_DIR`. |
| `/opt/litco-agent/toolenv` | A venv for the agent's own scripts: duckdb, pandas, matplotlib, python-docx, PyMuPDF, requests, openpyxl. It is first on the service's `PATH`, so `python` in the agent's terminal has these. |
| system packages | curl, git, jq, ripgrep, ffmpeg, 7zip, unrar, poppler-utils (`pdftotext`, `pdftoppm`), LibreOffice core/writer/calc, metric-compatible fonts, Node 24 LTS, Tailscale, ufw, fail2ban, unattended-upgrades. |
| `hermes` user | uid 10000, with linger enabled. Upstream's cron and Kanban workers use `systemd-run --user`, which needs hermes's own user manager. |

The unit is baked enabled and stopped, and no `/etc/litco-agent/env` exists in the image. `install-host.sh` fails if that file exists, and `build-image.sh` checks again before the snapshot.

## Versioning

The snapshot name is the version, and the version is a git tag on the fork.

1. Tag the commit you want on hosts, for example `git tag host-2026.09.27 && git push origin host-2026.09.27`.
2. Build: `deploy/host/build-image.sh --version host-2026.09.27 --ssh-key <fingerprint>`. The script refuses a version whose snapshot already exists, so a version always names one image.
3. Point the control plane at `litco-agent-host-host-2026.09.27`, and record that name on each host it creates.

`/opt/litco-agent/REF` on every host holds the exact commit it runs.

## How the control plane boots a host

The control plane builds a spec, renders user-data, and creates the droplet from the snapshot:

```
python3 deploy/host/render-user-data.py spec.json > user-data.yaml     # or: ... - < spec.json
```

It then calls DigitalOcean's droplet create with `image` set to the snapshot, `size` `s-4vcpu-8gb`, `region` `sfo3`, `monitoring` on, and `user_data` set to the output. The output is at most 64 KiB, which is DigitalOcean's limit; the renderer refuses anything larger. On a bad spec it exits 2 and names the field, and it never echoes a secret.

The A7 control plane is TypeScript. It can call this script, or port it. If it ports it, `tests/host/test_render_user_data.py` is the contract to keep passing.

### The spec

| Field | Required | Meaning |
|---|---|---|
| `matterId` | yes | The LitKit matter id. |
| `instanceUrl` | yes | The firm's instance, `https://<firm>.litco.ai`. |
| `hostSecret` | yes | Secret, at least 32 characters. LitKit sends it in `X-Host-Secret`, and it keys user assertions. |
| `agentToken` | yes | Secret. The matter-pinned LitKit token, `lkm_…`. |
| `modelProvider` | yes | A Hermes provider slug: `anthropic`, `openrouter`, `custom`, and so on. |
| `egressPolicy` | yes | `open` or `allowlist`. |
| `matterHome` | no | Default `/home/hermes/matter`. |
| `model` | no | Model name, such as `anthropic/claude-opus-4.6`. |
| `modelBaseUrl` | no | OpenAI-compatible base URL, used with provider `custom` (for example, a future LitKit model proxy). |
| `modelKey` | no | Secret. It lands in the provider's own key variable (`ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, …). For `custom` it lands in `LITCO_MODEL_API_KEY`. |
| `tailscaleAuthKey` | no | Secret. With it, the host joins the tailnet with Tailscale SSH. |
| `hostname` | no | `matter-<shortid>`. By default it is derived from the first eight letters and digits of `matterId`. |
| `egressAllowlist` | no | `[{"cidr": "203.0.113.0/24", "port": 443, "proto": "tcp"}]`. Allowed only with `egressPolicy: "allowlist"`. |
| `approvalsMode` | no | Empty (the upstream default), `manual`, `smart`, or `off`. See below. |
| `turnHost`, `turnPort` | no | Default `0.0.0.0` and `8765`. The firewall admits inbound traffic only on `tailscale0`. |
| `channelEnv` | no | Native Slack or Telegram settings, limited to `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_ALLOWED_USERS`, `SLACK_HOME_CHANNEL`, `SLACK_IGNORED_CHANNELS`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_USERS`, `TELEGRAM_GROUP_ALLOWED_USERS`, `TELEGRAM_GROUP_ALLOWED_CHATS`. |

### First boot, in order

1. cloud-init writes `/etc/litco-agent/env` (root:root, 0600), `/etc/litco-agent/matter.json` (non-secret), and, if a Tailscale key was given, `/etc/litco-agent/tailscale-authkey` (0600).
2. It sets the hostname to `matter-<shortid>` and creates the matter home, owned by `hermes`.
3. With a key, it runs `tailscale up --ssh --hostname=matter-<shortid> --auth-key=file:…` and then shreds the key file.
4. It resets ufw: deny all inbound, allow inbound on `tailscale0`, allow all outbound, plus the allowlist block when requested.
5. It runs `systemctl restart litco-agent.service`. It uses restart because the unit is baked enabled; the unit's `ConditionPathExists=/etc/litco-agent/env` kept it from running before this point.
6. The unit's `ExecStartPre` renders the profile, and the gateway starts with `litco_turn` on port 8765.

## The environment contract

systemd reads `/etc/litco-agent/env` as root for the unit's `EnvironmentFile=`. The `hermes` user cannot read the file. The gateway and the tools Hermes spawns get the values as environment variables.

| Variable | Secret | Read by |
|---|---|---|
| `LITCO_MATTER_ID` | no | turn server, init |
| `LITCO_INSTANCE_URL` | no | turn server, init, LitKit tools |
| `LITCO_MATTER_HOME` | no | turn server, init |
| `LITCO_TURN_HOST`, `LITCO_TURN_PORT` | no | turn server, init, drain |
| `LITCO_MODEL_PROVIDER`, `LITCO_MODEL`, `LITCO_MODEL_BASE_URL` | no | init (into `config.yaml`) |
| `LITCO_MODEL_KEY_ENV` | no | init. It is set to `LITCO_MODEL_API_KEY` only for provider `custom` with a key, and becomes `model.key_env`. |
| `LITCO_APPROVALS_MODE` | no | init |
| `LITCO_EGRESS_POLICY` | no | operators (informational) |
| `LITCO_HOST_SECRET` | yes | turn server |
| `LITCO_AGENT_TOKEN` | yes | turn server, LitKit tools |
| `<PROVIDER>_API_KEY` or `LITCO_MODEL_API_KEY` | yes | Hermes provider resolution |
| `SLACK_*`, `TELEGRAM_*` | yes | Hermes native adapters |

Two rules hold throughout. No script in this repo writes a secret; only cloud-init does, from user-data. And `litco-agent-init` reads only the variables in its `NON_SECRET_KEYS` list, so a template cannot pull in a secret: an unknown placeholder is an error. The tests check both.

Like the LitKit worker fleet, secrets ride in user-data, and DigitalOcean's metadata API serves user-data to processes on the droplet for its lifetime. The droplet serves one matter, and the agent on it already holds these secrets in its environment.

## Egress policy

`open` is the pilot posture: nothing inbound except over the tailnet, and all outbound traffic allowed (pilot plan section 7).

`allowlist` sets outbound to deny by default. It then allows the tailnet interface, DNS, NTP, Tailscale's direct UDP port 41641, and each `egressAllowlist` entry. ufw matches addresses, not names. Destinations behind DNS names (the model provider, the LitKit instance behind a CDN, Tailscale's control plane and relays, Westlaw and Lexis) need their current address ranges or a logging proxy. Without Tailscale's control-plane ranges, the host drops off the tailnet. The allowlist block is scaffolding for the day the pilot shows which destinations the agent needs; no host has run with it.

## Approvals: one knob, the owner decides

Hermes gates commands it classifies as dangerous. This image adds no override of its own. The single knob is `approvalsMode` in the spec, which becomes `LITCO_APPROVALS_MODE` and then `approvals.mode` in `config.yaml`:

| Value | Effect |
|---|---|
| empty (default) | No `approvals` block is rendered, so the upstream Hermes default applies (`smart` at fork time: a guardian model approves low-risk commands and escalates the rest). |
| `manual` | Every flagged command needs a human. |
| `smart` | The same as the upstream default, pinned. |
| `off` | No approval gate. This is the same as Hermes's `--yolo`. |

A turn that arrives through `litco_turn` has no approval channel. The turn server registers no approval notifier, and LitKit has no `/approve` surface. So when a command needs a human, the tool returns a "pending approval" result to the agent at once, and the agent sees that the command did not run. The turn does not hang for the approval timeout. (This comes from reading `tools/approval.py`: with no gateway notifier and no CLI, the gate returns `_pending_result`. The smoke test does not exercise it.) The native Slack and Telegram adapters, when configured, keep their own `/approve` flow.

Which posture a matter runs under is the owner's decision. The image ships the upstream default until the owner sets the knob.

## Stopping and restarting never kills a running turn

`systemctl stop` or `restart` first runs `litco-agent-drain`, which polls `GET /health` until `activeTurns` is 0. `TimeoutStopSec=infinity` means systemd waits as long as that takes. Then Hermes's stop marker runs, and `KillMode=mixed` sends SIGTERM to the gateway process alone. This mirrors the LitSpace worker rule. The drain counts turns on the turn server only. Native Slack and Telegram turns and cron jobs are drained by Hermes on SIGTERM. To stop with a turn still running, on purpose, use `systemctl kill litco-agent`. `LITCO_DRAIN_MAX_SECONDS` in the env file caps the wait if the owner ever wants a cap.

## Rolling the fork forward

New hosts: tag, rebuild the snapshot, and point the control plane at the new name. Hosts created from then on get the new code.

Existing hosts: update in place over Tailscale SSH, one host at a time. The restart waits for running turns.

```
ssh root@matter-<shortid>     # Tailscale SSH
REF=host-2026.10.04
cd /opt/litco-agent/app
git fetch --tags origin && git checkout --detach "$REF" && git rev-parse HEAD > /opt/litco-agent/REF
UV_PYTHON_INSTALL_DIR=/opt/litco-agent/python UV_PROJECT_ENVIRONMENT=/opt/litco-agent/app/.venv \
  uv sync --frozen --no-dev --extra messaging
install -m 0755 deploy/host/litco-agent-init /usr/local/bin/litco-agent-init
install -m 0755 deploy/host/litco-agent-drain /usr/local/bin/litco-agent-drain
install -m 0644 deploy/host/litco-agent.service /etc/systemd/system/litco-agent.service
systemctl daemon-reload && systemctl restart litco-agent
curl -s http://127.0.0.1:8765/health
```

The `uv sync` step is needed only when `uv.lock` changed, and it is harmless otherwise. A change to `install-host.sh` itself, such as a new system package, reaches existing hosts only by hand or by replacing the host from the new snapshot. Replacing a host means snapshotting the matter home first, as the control plane's idle path already does.

## The browser on Ubuntu 24.04

Ubuntu 24.04 restricts unprivileged user namespaces through AppArmor, so Chromium's own sandbox cannot start from the PM store path. Hermes detects this and launches Chromium with `--no-sandbox` (`tools/browser_tool_session.py`). That fits the permissive pilot posture. A later hardening step would be an AppArmor profile that grants `userns` to that binary, plus `AGENT_BROWSER_ARGS` set so that Hermes stops passing `--no-sandbox`.

## Testing locally

```
.venv/bin/python -m pytest tests/host -q        # or: make -C deploy/host test
deploy/host/build-image.sh --version v0 --dry-run
deploy/host/smoke.sh                             # needs a Docker daemon
```

`smoke.sh` exports the working tree, builds `smoke/Dockerfile` (Ubuntu 24.04 running `install-host.sh --container`, which skips Tailscale, ufw, fail2ban, unattended-upgrades, and linger), and boots systemd in the container. It then:

1. runs `systemd-analyze verify` on the unit;
2. checks that the unit is enabled and does not start without an env file;
3. writes a fake env file as cloud-init would and restarts the unit;
4. waits for `/health`;
5. checks the rendered profile, and checks that no secret reached `$HERMES_HOME` and that `hermes` cannot read the env file;
6. posts one `/turn` and asserts the SSE stream ends in a `final` frame with the stub model's answer;
7. renders a page with the PM Chromium as `hermes`;
8. stops the unit and checks that the drain ran.

The model is a stub OpenAI-compatible server inside the container (`smoke/stub_model.py`), so the turn runs through the real gateway, the `litco_turn` platform, `HermesTurnRunner`, and `AIAgent`. The unit tests in `tests/litco/` already cover the turn server with the fake runner.

## Not verified, because no droplet was created

- `build-image.sh` has never run against DigitalOcean. Its doctl flags, the SSH wait, and the snapshot and delete sequence are checked only as a dry-run plan.
- `install-host.sh` has run only in an arm64 Ubuntu 24.04 container with `--container`. The droplet-only steps have never run: the Tailscale install, `loginctl enable-linger`, fail2ban, unattended-upgrades, the sshd drop-in with `sshd -t`, persistent journald, and the x86_64 package set.
- cloud-init on a real droplet has not run. That covers `write_files` with `b64`, the hostname, `tailscale up --auth-key=file:` (the `file:` form needs a recent Tailscale), ufw on a real kernel, and the restart ordering. The rendered YAML is parsed in tests, not by cloud-init.
- `cloud-init clean --logs --machine-id` before the snapshot, and whether droplets from the snapshot then run their user-data, is unverified. That exact failure made LitKit's first worker image boot useless in July 2026.
- The allowlist egress policy has never been applied anywhere.
- Linger and the `user@10000.service` ordering: in the container the user manager fails to start (no logind), and the unit runs anyway. Cron jobs that need `systemd-run --user` are untested.
- A real model provider. The smoke turn used the stub model, not a provider key.
- The browser tool end to end through a model's tool call. The smoke renders a page with the same Chromium binary but does not drive `agent-browser` from a turn.
- The pending-approval behavior on `litco_turn`, which comes from reading the code.
- Native Slack and Telegram through `channelEnv`.
- The 100 GB block volume for productions (pilot section 5) is not attached or mounted by this tooling. The control plane or a later cloud-init block has to do that.
