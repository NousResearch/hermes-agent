---
name: agent-browser-takeover
description: Use when the owner must take over a headed agent browser over WireGuard noVNC, Grok Bot-style, including a Hermes-agent host bootstrap and optional egress through another WireGuard peer.
version: 0.3.0
author: TotalLag, Hermes Agent
license: MIT
platforms: [linux]
required_commands: [Xvfb, x11vnc, websockify, python3]
metadata:
  hermes:
    tags: [browser, vnc, novnc, wireguard, takeover, camoufox, socks]
    related_skills: []
    requires_toolsets: [terminal]
    config:
      - key: takeover.bind_ip
        description: VPS WireGuard IPv4 for noVNC. Never 0.0.0.0.
        default: ""
        prompt: WireGuard address to bind noVNC
      - key: takeover.wg_iface
        description: WireGuard interface that owns BIND_IP
        default: wg0
        prompt: WireGuard interface name
---

# Agent browser takeover (WireGuard noVNC)

Give the human owner live view and control of the agent's headed browser, the way Grok Bot hands over the bot screen. The owner types credentials on their own keyboard. Secrets never enter the agent context.

This is a mirror of an existing Xvfb framebuffer. It is not a second browser and not a virtual desktop.

From-scratch walkthrough: `references/fresh-vps.md`.
Hermes agent bootstrap: `references/hermes-agent-install.md`.
Fake lab topology: `references/topology.md`.
Why these binds: `references/why.md`.
Optional peer SOCKS egress: `references/peer-egress.md`.

## When to Use

- Agent hit a login, captcha, or MFA wall.
- Owner should drive the same Camoufox/Firefox window over VPN.
- You are putting this on a new VPS that already has WireGuard.
- Browser public traffic should leave via another WireGuard peer (home NAS, travel router, second box) instead of the VPS WAN.

Do not use this skill to scrape pages or expand comments.

## Architecture

```
agent browser (headed) → Xvfb :98
                       → x11vnc listen 127.0.0.1:5900
                       → websockify  BIND_IP:6080 → 127.0.0.1:5900
                       → owner opens http://BIND_IP:6080/vnc.html on the VPN

optional: browser HTTP(S) → socks5://PEER_WG_IP:1080  (other WG peer)
```

`BIND_IP` is this VPS WireGuard address. `PEER_WG_IP` is the other peer. Neither is shipped in this tree.

## Hermes agent install

`hermes skills install official/devops/agent-browser-takeover` copies this skill only. It does **not** install Xvfb, noVNC, or Camoufox.

When the user asks you to set this up:

1. `skill_view` this skill. `$SKILL_DIR` is the directory it reports.
2. `"$SKILL_DIR/scripts/install.sh" check` — no sudo. Read `MISSING=` / `READY=`.
3. If `wg` or `bind_ip` is missing, stop. Do not invent a mesh. Ask for the wg0 IPv4 and write `~/.hermes/takeover/env`.
4. If packages or Camoufox are missing, tell the user you need sudo once, then `"$SKILL_DIR/scripts/install.sh" vps`.
5. `"$SKILL_DIR/scripts/verify.sh"` must pass before you hand out `http://$BIND_IP:6080/vnc.html`.

Do not apt-get merely because the skill loaded. Full procedure: `references/hermes-agent-install.md`.

The owner URL is always `http://$BIND_IP:6080/vnc.html` on WireGuard. Fake mesh: `10.13.37.1` VPS / `10.13.37.4` peer (`references/topology.md`). Do **not** put noVNC on `127.0.0.1` — that is only raw x11vnc `:5900`. No WireGuard yet: `scripts/lab-dummy-iface.sh` then the same `install.sh vps` path. Copy `templates/lab.env.example` → `~/.hermes/takeover/env` and replace those addresses if they are not yours.

## Hard rules

- Bind noVNC to `BIND_IP` only. Never `0.0.0.0`.
- Bind raw RFB to `127.0.0.1` only. websockify must target `127.0.0.1:5900`, not `localhost:5900` (`localhost` can be `::1` and Connect fails).
- `-nopw` is acceptable only while RFB is loopback and noVNC is VPN-only.
- Port-up or `vnc.html` HTTP 200 is not success. Prove the RFB banner and a non-black framebuffer.
- Playwright pages die when the client disconnects. Hold one connected page or the VNC canvas is black even when x11vnc is healthy.
- Peer SOCKS binds that peer's WireGuard IP only. Do not change the VPS default route.

## From scratch (VPS)

Human or agent, same commands. `$SKILL_DIR` after a hub install is `$HERMES_HOME/skills/devops/agent-browser-takeover`.

```bash
"$SKILL_DIR/scripts/install.sh" check
mkdir -p ~/.hermes/takeover
cp -n "$SKILL_DIR/templates/env.example" ~/.hermes/takeover/env
# set BIND_IP to `ip -4 -o addr show dev wg0`
"$SKILL_DIR/scripts/install.sh" vps
"$SKILL_DIR/scripts/verify.sh"
```

Owner URL: `http://$BIND_IP:6080/vnc.html`

## Optional peer egress

On the other WireGuard host, set `PEER_WG_IP` and run `"$SKILL_DIR/scripts/install.sh" peer`.
On the VPS, set `HERMES_BROWSER_PROXY=socks5://PEER_WG_IP:1080` and restart `camoufox-server`.
Details: `references/peer-egress.md`.

## Login handoff

1. Agent drives until login/captcha/MFA. Pause. Do not guess credentials or trigger a second code.
2. Send the owner the VPN viewer URL. They click the canvas and type.
3. Keep the same Playwright connection alive. A new `connect()` drops the session.

VNC is for the owner. Agent does not click names or chrome on the canvas.

## Common Pitfalls

1. websockify → `localhost:5900` while x11vnc is IPv4 loopback: Connect refused. See `references/novnc-connect-refused.md`.
2. Healthy noVNC + black screen: no live Playwright page. Hold one. See `references/black-vnc-empty-page.md`.
3. Binding noVNC or SOCKS to `0.0.0.0`.
4. Expecting `ssh -D` through the peer (`AllowTcpForwarding no` is common).
5. Direct `Camoufox()` without `proxy=` while the server uses peer SOCKS — leaks the VPS WAN.

## Privacy

No host mesh IPs, cookie DBs, or live session URLs belong in this tree. Fill `~/.hermes/takeover/env` on the box. See `references/privacy.md`.

## Verification Checklist

- [ ] `python3 "$SKILL_DIR/scripts/test_takeover.py"` passes
- [ ] `install.sh check` runs without sudo
- [ ] `BIND_IP=0.0.0.0` is rejected
- [ ] `scripts/verify.sh` on a live box: RFB loopback, noVNC on `BIND_IP`, Camoufox `/json/version` 200
- [ ] Hold process prints `HOLDING` and the VNC capture is not black
- [ ] If `HERMES_BROWSER_PROXY` is set, ipify via SOCKS differs from the VPS WAN
