# Black noVNC canvas with Connect succeeding

If the owner can Connect but the canvas is black, do **not** treat it as
the websockify/`localhost:5900` bind bug. That bug is Connect refused
plus journal `ECONNREFUSED`. See `novnc-connect-refused.md`.

## Proven fault

- Camoufox `/json/version` 200, Xvfb `:98` ok, Connect worked
- VNC capture ~100% black
- Cause: no Playwright client holding a page. The server process stays
  up; pages/contexts are destroyed on client disconnect.

## Fix

Hold one `firefox.connect($HERMES_BROWSER_WS)` session for the whole
takeover:

```bash
systemctl --user restart camoufox-hold
# journal should contain HOLDING plus a title/URL
```

Script: `scripts/hold_page.py`. Use the same Python as the Camoufox
server (`~/.hermes/takeover/venv/bin/python`). Do not use a throwaway
background shell as the holder.

Default hold URL is `https://example.com/` (gray `rgb(238,238,238)`).

## Verify before telling the owner to refresh

1. `camoufox-hold` journal contains `HOLDING`
2. Capture `127.0.0.1::5900`: non-black ≳15%, top color matches the
   held page
3. Refresh Connect on `http://BIND_IP:6080/vnc.html`
