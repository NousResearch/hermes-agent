---
name: surfshark-vpn-exit
description: Route agent egress through a WireGuard SOCKS proxy.
version: 0.1.0
author: Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [VPN, WireGuard, SOCKS, Proxy, Egress, systemd]
    category: devops
    related_skills: [pinggy-tunnel]
---

# Surfshark VPN Exit Skill

Route a host's outbound traffic through a Surfshark WireGuard tunnel exposed as
a local SOCKS5 proxy (via `wireproxy`), so the agent and gateway egress with the
VPN provider's IP/ASN instead of the datacenter's. This skill covers the
userspace proxy, pointing services at it, verifying the public IP actually
changed, and making it survive reboots. It does **not** set up kernel-level
`wg-quick` routing, manage a Surfshark account, or guarantee any model API /
OAuth endpoint is reachable through the VPN — those depend on the provider.

## When to Use

- A VPS / cloud host needs its outbound IP to read as a consumer VPN endpoint
  rather than a datacenter ASN (geo-restricted APIs, region testing, scraping).
- You want per-process egress routing without root-level `wg-quick`, network
  namespaces, or touching the host routing table.
- The agent or gateway should send traffic through the tunnel via
  `HTTPS_PROXY` / `ALL_PROXY`, while leaving localhost and control planes direct.
- You need the proxy to come back automatically after a reboot.

If you only need to expose a local port publicly (not route egress), use the
`pinggy-tunnel` skill instead.

## Prerequisites

- **Linux with systemd** and user-level lingering support (`loginctl`).
- **`wireproxy`** on `PATH` — a userspace WireGuard client that serves SOCKS5
  (https://github.com/pufferffish/wireproxy). No kernel module or root needed.
- **A Surfshark WireGuard config** (private key + peer endpoint + allowed IPs),
  generated from the Surfshark dashboard. Store it at a path you control, e.g.
  `~/.config/wireproxy/exit.conf`. Treat the private key as a secret — never
  commit it.
- **`curl`** for verification. `jq` is optional (the scripts fall back to grep).

No `HERMES_*` env vars are required. All tunable values are passed to the helper
script as plain env vars (`SOCKS_PORT`, `WIREPROXY_CONF`, `VERIFY_URL`, …) or
written into the rendered systemd units.

## How to Run

Drive everything through the `terminal` tool. The helper script
`scripts/vpn-exit.sh` wraps the full lifecycle; read it with `read_file` before
running so you can see the exact commands. Typical flow:

```bash
SKILL_DIR=skills/devops/surfshark-vpn-exit          # adjust to the installed path
WIREPROXY_CONF=~/.config/wireproxy/exit.conf SOCKS_PORT=1080 \
  bash "$SKILL_DIR/scripts/vpn-exit.sh" install-service
bash "$SKILL_DIR/scripts/vpn-exit.sh" up
bash "$SKILL_DIR/scripts/vpn-exit.sh" verify        # FAILS if egress IP did not change
bash "$SKILL_DIR/scripts/vpn-exit.sh" persist       # enable on boot + linger
```

`verify` is the load-bearing step: it compares the direct public IP with the
proxied public IP and exits non-zero if they match, so a silent fallback to the
datacenter IP is treated as a failure, never a success.

## Quick Reference

```bash
# 1. Render + install the user-level wireproxy service from the template
WIREPROXY_CONF=~/.config/wireproxy/exit.conf SOCKS_PORT=1080 \
  bash scripts/vpn-exit.sh install-service

# 2. Start / stop / status
bash scripts/vpn-exit.sh up
bash scripts/vpn-exit.sh down
bash scripts/vpn-exit.sh status

# 3. Confirm the public IP/ASN now reads as the VPN, not the datacenter
bash scripts/vpn-exit.sh verify
curl --proxy socks5h://127.0.0.1:1080 -s https://ipinfo.io/json

# 4. Point a systemd-managed service (e.g. the gateway) at the proxy
bash scripts/vpn-exit.sh wire-service hermes-gateway

# 5. Survive reboots
bash scripts/vpn-exit.sh persist
```

## Procedure

### 1. Bring up the SOCKS proxy

`install-service` renders `templates/wireproxy.service.template` into
`~/.config/systemd/user/<service>.service` (default `hermes-wireproxy`) pointing
at your `WIREPROXY_CONF`. The WireGuard config must contain a `[Socks5]` section
binding `127.0.0.1:<SOCKS_PORT>` — see `templates/wireproxy.conf.template` for
the shape. Then `up` runs `systemctl --user start` and waits for the port to
listen.

### 2. Verify the exit IP actually changed

Run `verify`. It fetches `VERIFY_URL` (default `https://ipinfo.io/json`) twice —
once directly and once through `socks5h://127.0.0.1:<port>` — and compares the
`ip`/`org` fields. Use `socks5h` (not `socks5`) so DNS resolves through the
tunnel and does not leak. If the two IPs are equal, the proxy is not routing and
the script exits non-zero. Never report success on an unverified tunnel.

### 3. Point the agent / gateway egress at the proxy

Egress is controlled with standard proxy env vars. For a systemd-managed Hermes
service, `wire-service <unit>` writes a drop-in at
`~/.config/systemd/user/<unit>.service.d/10-proxy.conf` from
`templates/proxy-env.conf.template`, setting:

```ini
[Service]
Environment=HTTPS_PROXY=socks5h://127.0.0.1:1080
Environment=ALL_PROXY=socks5h://127.0.0.1:1080
Environment=NO_PROXY=127.0.0.1,localhost,::1
```

Then `systemctl --user daemon-reload && systemctl --user restart <unit>`. Keep
`NO_PROXY` covering loopback so health checks and the JSON-RPC control plane
stay direct.

### 4. Failover

If `verify` fails or the tunnel drops, decide explicitly: either retry the
tunnel, switch to a backup Surfshark endpoint (a second `WIREPROXY_CONF`), or
stop egress entirely. Do **not** let traffic silently fall through to the
datacenter IP — clear the proxy env (`down` + restart the service unproxied)
only as a deliberate, logged choice.

### 5. Persistence

`persist` runs `systemctl --user enable <service>` and
`loginctl enable-linger "$USER"` so the user service starts at boot without an
interactive login session.

## Pitfalls

- **Silent fallback to the datacenter IP is the #1 failure.** If the SOCKS proxy
  is down, a misconfigured `HTTPS_PROXY` or a tool that ignores it will egress
  directly. Always gate on `verify`; treat "IP unchanged" as a hard failure.
- **Use `socks5h`, not `socks5`.** `socks5h://` resolves DNS through the tunnel;
  `socks5://` resolves locally and leaks your real DNS / location.
- **Keep loopback in `NO_PROXY`.** Routing `127.0.0.1`/`localhost` through the
  proxy breaks local health checks, the gateway's JSON-RPC control plane, and
  any sidecar on the same host.
- **VPN egress can break model APIs and OAuth.** Some provider/auth endpoints
  block or rate-limit known VPN ASNs. Verify the specific endpoints you depend
  on after enabling the proxy; scope the proxy to only the services that need it.
- **Telegram / messaging reachability.** Long-poll and webhook connections may
  behave differently from a VPN IP. Confirm the gateway can still reach its
  platform after switching egress.
- **Never commit the WireGuard private key.** It lives only in `WIREPROXY_CONF`
  on the host. The templates ship with placeholders, not real keys.
- **User services need linger.** Without `loginctl enable-linger`, a `--user`
  systemd service stops when the SSH session ends and won't start at boot.

## Verification

```bash
# Direct vs proxied public IP — must differ
DIRECT=$(curl -s https://ipinfo.io/json | grep -o '"ip": *"[^"]*"')
PROXIED=$(curl --proxy socks5h://127.0.0.1:1080 -s https://ipinfo.io/json | grep -o '"ip": *"[^"]*"')
echo "direct=$DIRECT proxied=$PROXIED"
test "$DIRECT" != "$PROXIED" && echo "OK: egress is routed" || echo "FAIL: same IP"

# Or let the script do the comparison and ASN/org print-out
bash scripts/vpn-exit.sh verify
```

Expected: the proxied `ip`/`org` reads as the VPN provider's endpoint, the
direct values read as the host's datacenter, and `vpn-exit.sh verify` exits `0`.
