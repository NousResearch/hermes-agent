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
