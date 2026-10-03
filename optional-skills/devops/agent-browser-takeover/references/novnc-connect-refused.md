# noVNC "Failed to connect to server"

The owner can load `vnc.html` (HTTP 200) and still fail **Connect**.
The HTML/JS is static; Connect is websockify → RFB.

## Proven fault

- x11vnc listened on the VPN IP, not `127.0.0.1:5900`
- websockify targeted `localhost:5900`
- Journal: `Failed to connect to localhost:5900: [Errno 111] Connection refused`
- `curl vnc.html` / `rfb.js` still 200 — that probe is insufficient

## Required bind

```
x11vnc  … -rfbport 5900 -listen 127.0.0.1 -localhost
websockify --web /usr/share/novnc  BIND_IP:6080  127.0.0.1:5900
```

Use `127.0.0.1`, never `localhost` (can resolve to `::1` and miss the
IPv4 socket). Do not bind raw VNC on the VPN IP just to make Connect
work.

## Connect-ready checks (all required)

1. `ss`: `127.0.0.1:5900` listen; `BIND_IP:6080` listen; **no** VPN `:5900`
2. TCP to `127.0.0.1:5900` returns `RFB 003.008`
3. WebSocket `ws://BIND_IP:6080/websockify` also returns `RFB 003.008`
4. Pixel/content check against `127.0.0.1::5900` before claiming the
   mirror is the bot screen

If Connect succeeds and the canvas is still black, that is an empty
page (no held Playwright client), not this bind bug. See
`black-vnc-empty-page.md`.
