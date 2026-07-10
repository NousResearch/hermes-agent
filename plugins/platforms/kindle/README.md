# Kindle Scribe platform

The Kindle platform connects a handwriting-oriented web bridge to the normal
Hermes Gateway message pipeline:

```text
Kindle browser -> diary/OCR bridge -> authenticated localhost ingest -> Hermes Gateway
```

This adapter deliberately does not expose Hermes to the LAN or internet. It
listens on `127.0.0.1`, authenticates the companion bridge with
`KINDLE_INGEST_TOKEN`, applies the configured Hermes user allowlist, and turns
each accepted note into a standard `MessageEvent`. Models, memory, sessions,
skills, and `platform_toolsets.kindle` then work exactly as they do for other
messaging platforms.

## Companion bridge

The reference handwriting UI, two-stage OCR bridge, and deployment guide are in
[hermes-agents-guide-to-the-galaxy](https://github.com/lEWFkRAD/hermes-agents-guide-to-the-galaxy).

The bridge supports:

- Full-page Kindle Scribe pen input.
- Vision OCR followed by an optional text-model proper-name cleanup pass.
- Stable Hermes thread IDs for new and continued notebook entries.
- Artifact/image/HTML annotation workspaces.
- Passwordless trusted-LAN operation or explicit browser authentication.
- Away-from-LAN access through Tailscale Funnel with a permanent, high-entropy
  bookmark key that does not depend on Kindle cookies or local storage.

## Gateway configuration

Set a random bridge-to-adapter secret and restrict the channel to known users:

```powershell
$env:KINDLE_INGEST_TOKEN = '<random-secret>'
$env:KINDLE_ALLOWED_USERS = 'jeff'
```

Configure the tools this platform may use:

```yaml
platform_toolsets:
  kindle:
    - memory
    - web
```

Then start the Gateway and verify `http://127.0.0.1:8793/health` before starting
the companion bridge.

## Remote-access security boundary

Do not expose port `8793`. Remote browser access terminates at the companion
bridge; the bridge remains the only caller of this adapter. If using Tailscale
Funnel, configure `DIARY_REMOTE_KEY` in the companion bridge before enabling the
public HTTPS route. Remote requests without that key must fail before they can
reach this adapter, session history, or stored handwriting.
