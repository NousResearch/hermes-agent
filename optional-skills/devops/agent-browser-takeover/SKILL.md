---
name: agent-browser-takeover
description: Use when the owner must take over a headed agent browser over WireGuard noVNC, Grok Bot-style, to type login, captcha, or MFA themselves.
version: 0.1.0
author: TotalLag, Hermes Agent
license: MIT
platforms: [linux]
metadata:
  hermes:
    tags: [browser, vnc, novnc, wireguard, takeover, camoufox]
    related_skills: []
---

# Agent browser takeover (WireGuard noVNC)

Give the human owner live view and control of the agent's headed browser, the way Grok Bot hands over the bot screen. The owner types credentials on their own keyboard. Secrets never enter the agent context.

This is a mirror of an existing Xvfb framebuffer. It is not a second browser and not a virtual desktop.

## When to Use

- Agent hit a login, captcha, or MFA wall.
- Owner should drive the same Camoufox/Firefox (or other headed) window over VPN.
- You need to expose noVNC only on a WireGuard address, never on a public NIC.

Do not use this skill to scrape pages or expand comments.

## Architecture

```
agent browser (headed) → Xvfb :98
                       → x11vnc listen 127.0.0.1:5900
                       → websockify  <wireguard-ip>:6080 → 127.0.0.1:5900
                       → owner opens http://<wireguard-ip>:6080/vnc.html on the VPN
```

## Hard rules

- Bind noVNC to the WireGuard IP only. Never `0.0.0.0`.
- Bind raw RFB to `127.0.0.1` only. websockify must target `127.0.0.1:5900`, not `localhost:5900` (`localhost` can be `::1` and Connect fails).
- `-nopw` is acceptable only while RFB is loopback and noVNC is VPN-only. Broader exposure needs a password.
- Port-up or `vnc.html` HTTP 200 is not success. Prove the RFB banner and a non-black framebuffer.
- Playwright pages die when the client disconnects. Hold one connected page or the VNC canvas is black even when x11vnc is healthy.

## Prerequisites

```bash
sudo apt-get install -y x11vnc websockify novnc
# optional: pip install vncdotool
```

A headed browser already rendering on the Xvfb display (Camoufox on `:98` is the usual case).

## How to Run

`SKILL_DIR` is this skill directory. Set `BIND_IP` to the host WireGuard address.

```bash
export BIND_IP="$(ip -4 -o addr show dev wg0 | awk '{print $4}' | cut -d/ -f1)"
# or: export BIND_IP=10.x.y.z
"$SKILL_DIR/templates/takeover_view.sh" start
"$SKILL_DIR/templates/takeover_view.sh" status
```

Keep a page on screen so the mirror is not black:

```bash
/usr/bin/python3 "$SKILL_DIR/scripts/hold_page.py"
```

Owner URL: `http://$BIND_IP:6080/vnc.html`

Stop with `"$SKILL_DIR/templates/takeover_view.sh" stop`.

## Login handoff

1. Agent drives until login/captcha/MFA. Pause. Do not guess credentials or trigger a second code.
2. Send the owner the VPN viewer URL. They click the canvas and type.
3. Keep the same Playwright connection alive. A new `connect()` drops the session.

VNC is for the owner. Agent does not click names or chrome on the canvas.

## Verify before handing the URL

1. `ss` shows RFB on `127.0.0.1:5900` only, noVNC on `$BIND_IP:6080`, nothing on `0.0.0.0:6080`.
2. WebSocket to `ws://$BIND_IP:6080/websockify` returns `RFB 003.008`.
3. Hold `https://example.com/` (gray `rgb(238,238,238)`). Capture `127.0.0.1::5900`. Non-black pixels ≳15% and top color matches.

## Common Pitfalls

1. websockify → `localhost:5900` while x11vnc is IPv4 loopback: Connect refused.
2. Binding x11vnc to the VPN IP and noVNC to localhost: same failure.
3. Healthy noVNC + black screen: no live Playwright page. Hold one. Do not restart x11vnc for that.
4. Binding noVNC to `0.0.0.0` or a public DNS name.

## Privacy

No host mesh IPs, cookie DBs, or live session URLs belong in this tree. `BIND_IP` is the installer's WireGuard address.

## Verification Checklist

- [ ] `python3 "$SKILL_DIR/scripts/test_takeover.py"` passes
- [ ] `BIND_IP=0.0.0.0` is rejected by the launcher
- [ ] RFB is loopback; noVNC is WireGuard-only
- [ ] Hold process prints `HOLDING` and the VNC capture is not black
