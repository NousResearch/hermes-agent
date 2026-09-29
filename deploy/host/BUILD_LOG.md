# Host image build log

## 2026-09-28: `litco-agent-host-2026.09.28-rc1` (first cloud run)

| Item | Value |
|---|---|
| Snapshot | `litco-agent-host-2026.09.28-rc1`, id `247382759`, 6.39 GB, min disk 80 GB, region `sfo3`, status available |
| litco-agent ref | `9b5de925bd0a2c4dcf4f0604084e56aa83129bbe` on branch `image/2026-09-28` (PR #4 `fix/e2e-defects` + PR #5 `fix/soul-deliverables`, which include #1, #2, #3), plus the build-script fix below |
| Command | `deploy/host/build-image.sh --version 2026.09.28-rc1 --ref 9b5de925… --size s-2vcpu-4gb --region sfo3 --ssh-key 56833384` |
| Builder | droplet `604235081` (`litco-agent-builder-2026-09-28-rc1`, `s-2vcpu-4gb`, tag `litco-host-builder`). Created about 03:42:20Z and deleted at 03:50:19Z, so it lived about 8 minutes. The install took about 6 minutes and the snapshot 71 seconds. |
| Verify droplet | droplet `604236423` (`litco-host-verify`, same size and tag), created from the snapshot with no user-data. It was created at 03:50:55Z and deleted at 03:51:52Z by an EXIT trap. |
| Cost | Droplet time was about 9 minutes at $0.0357/h, under $0.01. The snapshot costs 6.39 GB × $0.06/GB-month, about $0.38 a month. |
| Pre-existing droplets | yavarlaw, litkit-prod-1, wispr-proxy, litlex-prod-1 and alice were the same set, by id, before and after. No volume, DNS record, firewall or Tailscale node was created. |
| Pre-flight | `pytest tests/host tests/litco` passed 128 tests with 1 skip (`systemd-analyze` is absent on macOS). The dry-run plan was read before the real run. |

### Fix made before the run

`build-image.sh` changes:

- It tags the builder `litco-host-builder` by default, and a new `--tag` option overrides that.
- The EXIT trap now also fires on INT, TERM and HUP, so a ^C still deletes the builder.
- If `droplet create` dies before it returns an id, the trap finds the builder by tag and exact name and deletes it. This path was tested with a fake doctl, and the trap deleted only the exact builder name.
- The SSH calls send keepalives.

The script needed no fixes during the run. The first attempt succeeded.

### What was verified on the verify droplet (booted from the snapshot, no user-data)

- **cloud-init ran fresh.** DigitalOcean injected the SSH key, the hostname became `litco-host-verify`, and the machine id was new. That shows `cloud-init clean --logs --machine-id` before the snapshot worked, so droplets from this image will process their user-data.
- **The service stays down without secrets.** `litco-agent.service` is enabled but inactive, with `ConditionResult=no`. `/etc/litco-agent/env` and `/home/hermes/.hermes/.env` are both absent.
- **The image runs the pinned code.** `/opt/litco-agent/REF` holds `9b5de925…`.
- **The service user is in place.** `hermes` is uid 10000, and `Linger=yes`.
- **Python and its SQLite are correct.** The Hermes venv runs Python 3.14.7 with SQLite 3.53.1. The tool venv imports duckdb, pandas, python-docx and pymupdf.
- **The tools are installed.** Node v24.21.0, Tailscale 1.102.4 (installed, not joined), rg, ffmpeg, 7z, pdftotext and soffice. The PM store holds agent-browser 0.26.0 and chromium-1208.
- **Hardening services are up.** fail2ban and unattended-upgrades are active, `sshd -t` passes, and `systemctl --failed` is empty.
- **The firewall is off until first boot.** ufw is inactive in the image, and cloud-init enables it from user-data. Without user-data, SSH on the public IP worked.
- **Disk use is small.** The root disk has 6.0 GB used of 77 GB.

### Not verified

- **cloud-init user-data on a real matter host was not run.** That covers the env file written from b64, the `matter-<shortid>` hostname, the ufw reset to tailscale0-only, and the `systemctl restart` ordering. The gateway reaching `/health` on a droplet was not tested either.
- **The Tailscale join was not run.** Neither `tailscale up --auth-key=file:` nor Tailscale SSH was tried.
- **The monitoring agent was not confirmed.** `do-agent` showed `inactive` about one minute after boot, while cloud-init was still `running`. Whether monitoring reports later was not checked.
- **Cron jobs through `systemd-run --user` were not exercised.** Linger is on, but nothing ran through hermes's user manager.
- **No real model provider was used.** No turn ran, and neither the browser tool nor the approval path was tried.
- **The production size was not booted.** The snapshot's min disk is 80 GB, so it fits `s-2vcpu-4gb` and larger. The control plane's `s-4vcpu-8gb` should work, but that size was not tried.

## 2026-09-28: `litco-agent-host-2026.09.28-rc2` (rebuild from merged main)

rc2 replaces rc1. rc1 (image `247382759`) was deleted after rc2 passed its boot check.

| Item | Value |
|---|---|
| Snapshot | `litco-agent-host-2026.09.28-rc2`, id `247411090`, 6.39 GB, min disk 80 GB, region `sfo3`, status available |
| litco-agent ref | `74abe1a6ec42a41cc862908f7f5d73e99a38a89a`, the merge of PR #6 into `main` (PRs #1–#6). Its tree is identical to PR #6's reviewed head `3c6acae8`. |
| Command | `deploy/host/build-image.sh --version 2026.09.28-rc2 --ref 74abe1a6… --size s-2vcpu-4gb --region sfo3 --ssh-key 56833384` |
| Builder | droplet `604248615` (`litco-agent-builder-2026-09-28-rc2`, `s-2vcpu-4gb`, tag `litco-host-builder`). It ran from about 05:04Z to 05:13:11Z, about 9 minutes. The run exited 0 on the first attempt. |
| Verify droplet | droplet `604250550` (`litco-host-verify`, same size and tag), created from the snapshot with no user-data. It was created at 05:13:37Z and deleted at 05:15:38Z by an EXIT trap. |
| Cost | Droplet time was about 11 minutes at $0.0357/h, under $0.01. The rc2 snapshot costs about $0.38 a month. Deleting rc1 removes its equal charge. |
| Pre-existing droplets | yavarlaw, litkit-prod-1, wispr-proxy, litlex-prod-1 and alice were the same set, by id, before and after. No volume, DNS record, firewall or Tailscale node was created. The only image deleted was rc1. |
| Pre-flight | `pytest tests/host tests/litco` passed 135 tests with 1 skip. `ruff check` was clean. `hermes_cli.plugin_validate` passed on `plugins/litkit`. |

### What was verified on the verify droplet

The rc1 checks all passed again on rc2:

- cloud-init ran fresh, with a new machine id, the hostname applied and the SSH key injected.
- `litco-agent.service` is enabled but inactive, with `ConditionResult=no`, and no env file exists.
- `/opt/litco-agent/REF` holds `74abe1a6…`.
- `hermes` is uid 10000, and `Linger=yes`.
- The Hermes venv runs Python 3.14.7 with SQLite 3.53.1.
- The tool venv, Node 24.21, Tailscale 1.102.4, LibreOffice, the PM Chromium and agent-browser are present.
- fail2ban and unattended-upgrades are active, `sshd -t` passes, and no systemd units failed.

rc2 also closed two items that rc1 left open. This time the check waited for `cloud-init status --wait` to report `done` before inspecting the droplet.

- **The DigitalOcean agents run.** `do-agent` and `droplet-agent` are both active once first boot finishes. The `inactive` reading on rc1 was taken while cloud-init was still running.
- **journald is persistent.** `/var/log/journal/<machine-id>/system.journal` exists.

### Not verified

These items are unchanged from rc1:

- Matter user-data on a real host, meaning the env file, the hostname, the ufw reset, the restart, and the gateway's `/health` on a droplet.
- The Tailscale join.
- Cron jobs through `systemd-run --user`.
- A real model turn, the browser tool from a turn, and the approval path.
- Booting the `s-4vcpu-8gb` size.

## 2026-09-29: `litco-agent-host-2026.09.29-rc3` (Ana: thread context, actor, channel; litkit_channel_history)

rc3 adds PR #8 (`feat/ana-thread-context`): `actor` / `threadContext` / `litkitChannel` on `POST /turn`, Ana's framing in the turn prompt and SOUL.md, and the `litkit_channel_history` tool. rc2 (image `247411090`) is kept until rc3 passes a real-turn smoke on a re-provisioned matter host; delete it then.

| Item | Value |
|---|---|
| Snapshot | `litco-agent-host-2026.09.29-rc3`, id `247594520`, 6.26 GiB, region `sfo3`, status available |
| litco-agent ref | `da57a7de6d`, the merge of PR #8 into `main` |
| Command | `deploy/host/build-image.sh --version 2026.09.29-rc3 --ref da57a7de6d --size s-2vcpu-4gb --region sfo3 --ssh-key 56833384` (dry-run plan read first) |
| Builder | droplet `604732141` (`litco-agent-builder-2026-09-29-rc3`, `s-2vcpu-4gb`, tag `litco-host-builder`, 146.190.165.170). Ran 18:18:04Z → 18:24:28Z (about 6.5 minutes); shutdown 18:23:07Z, snapshot 18:23:18Z–18:24:25Z. Exit 0 on the first attempt; the builder was deleted by step 11 (no builder or verify droplet remains). |
| Cost | Under $0.01 of droplet time; the snapshot costs about $0.38 a month while it exists. |
| Pre-flight | `pytest tests/litco tests/host`: 150 passed, 1 skipped (`systemd-analyze`, macOS). `ruff check litco tests/litco` clean. |
| Not verified | No real turn ran from this image yet. The control plane still points at rc2 (`platform_agent_fleet`); switching it, re-provisioning the demo host and running the wave-3 smoke ("@Ana what did Jane ask for?" names Jane; `litkit_channel_history` tool pill visible) is the next step in the litkit-notes ledger. |
