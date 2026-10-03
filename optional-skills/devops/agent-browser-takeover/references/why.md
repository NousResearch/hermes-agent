# Why this shape (POC)

This optional skill is a **working POC** of Grok Bot-style owner takeover:
the human types login/captcha/MFA on the same headed browser the agent already
has open. We ran that on a VPS + WireGuard mesh. Addresses in this tree are
the fake lab in `topology.md`, not our live net.

## Decisions that survived contact with the bug

**Mirror the existing Xvfb, do not open a second browser.**
The owner must see the bot's tab, cookies and all. A second Playwright
browser is a logged-out clone. Screenshot streaming diverges from input.

**x11vnc + noVNC, not a custom RFB client.**
Phone/browser viewer, no native client. Hand-rolled RFB desynced on cursor
pseudo-rects. `vncdotool` / pixel checks for proof.

**Raw RFB on `127.0.0.1` only; noVNC on the WireGuard IP only.**
Standing VPS rule: nothing on `0.0.0.0` or the public NIC. Connect failed
when x11vnc listened on the VPN IP and websockify targeted `localhost:5900`
(`localhost` → `::1`, IPv4 socket missed, `ECONNREFUSED`). websockify must
target `127.0.0.1:5900`.

**No VNC password in this posture.**
RFB is loopback; noVNC is VPN-only. Password is friction without coverage.
Revisit if either bind changes.

**Hold one Playwright page.**
Connect can succeed on a black canvas. Camoufox `/json/version` 200 is not
a page. Playwright destroys pages on client disconnect. `hold_page.py` is
the mirror content, not a VNC restart.

**Peer SOCKS for browser *origin*, not a default-route change.**
Takeover listeners stay on the VPS. Public HTTP(S) from the headed browser
can leave via another WireGuard peer (home NAS / travel router / second
box). Do not shift the VPS default route (that would risk the VPN and
noVNC). SSH `-D` is not the path — peers often have `AllowTcpForwarding no`.
Run SOCKS on the peer, bound to that peer's wg address. Disable WebRTC so
the VPS WAN does not leak beside SOCKS.

**Skill copy ≠ host install.**
`hermes skills install` copies this directory. Xvfb/noVNC/Camoufox stay off
the default Hermes wheel (heavy, Linux, needs a VPN address). The agent
runs `install.sh check` then `vps` after BIND_IP and sudo are explicit.

## Out of scope for this POC

- Comment scraping or research briefs
- Persistent `user_data_dir` / saved logins across Camoufox restarts
- In-chat “hand me the controls” button (Grok Bot has it; we use a URL)
- Generating WireGuard keys or a mesh for you
