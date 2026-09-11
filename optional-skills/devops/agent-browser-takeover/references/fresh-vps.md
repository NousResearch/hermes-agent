# Fresh VPS, end to end

Goal: a new Ubuntu box with WireGuard already up can serve a headed agent
browser that the owner drives over noVNC on the VPN, the way Grok Bot
hands over the bot screen.

You need:

- Ubuntu 24.04 (or similar) VPS
- `wg0` (or `WG_IFACE`) with a private IPv4. The owner device is a peer
  on the same mesh. This skill does not generate keys.
- Ability to `sudo` for packages and `loginctl enable-linger`

Do **not** put noVNC on `0.0.0.0`, a public NIC, DNS, or a reverse proxy.

## 1. Get the skill onto the box

From a hermes-agent checkout (or after `hermes skills install` copies it):

```bash
SKILL_DIR=optional-skills/devops/agent-browser-takeover
mkdir -p ~/.hermes/takeover
cp "$SKILL_DIR/templates/env.example" ~/.hermes/takeover/env
```

## 2. Fill env

```bash
# VPS WireGuard address, never 0.0.0.0
BIND_IP=$(ip -4 -o addr show dev wg0 | awk '{print $4}' | cut -d/ -f1)
# edit ~/.hermes/takeover/env and set BIND_IP=$BIND_IP
```

Leave `HERMES_BROWSER_PROXY` empty until the optional peer SOCKS is up
(`references/peer-egress.md`).

## 3. Install

```bash
"$SKILL_DIR/scripts/install.sh" vps
"$SKILL_DIR/scripts/verify.sh"
```

`install.sh vps` will:

1. `apt-get` Xvfb, x11vnc, websockify, noVNC, Camoufox system libs
2. Create `~/.hermes/takeover/venv` and `python -m camoufox fetch` (~700MB)
3. Install user units: `camoufox-server`, `camoufox-hold`,
   `hermes-takeover-vnc`, `hermes-takeover-novnc`
4. Enable linger so they survive logout

Manual fallback without systemd: `"$SKILL_DIR/templates/takeover_view.sh" start`
plus `hold_page.py` after the Camoufox WS is up.

## 4. Owner takeover

1. Agent pauses on login / captcha / MFA. Do not type secrets into chat.
2. Owner, on a WireGuard client, opens `http://$BIND_IP:6080/vnc.html`
3. Click Connect, click the canvas, type.
4. Keep the same Playwright connection. A new `connect()` drops the session.

## Shape

```
Camoufox WS 127.0.0.1:9377
  optional public egress: socks5://PEER_WG_IP:1080
  renders on Xvfb :98
    x11vnc → 127.0.0.1:5900          loopback only
      websockify → BIND_IP:6080      WireGuard only
        owner: http://BIND_IP:6080/vnc.html
```

## Verify (do not skip)

`vnc.html` HTTP 200 is not success.

- RFB listens on `127.0.0.1:5900` only
- noVNC listens on `$BIND_IP:6080`, not `0.0.0.0`
- WebSocket / TCP banner is `RFB 003.008`
- Hold `https://example.com/` so the canvas is not black
- `verify.sh` covers the listeners, banner, and Camoufox `/json/version`

If Connect is refused, see `novnc-connect-refused.md`.
If Connect works but the canvas is black, see `black-vnc-empty-page.md`.
