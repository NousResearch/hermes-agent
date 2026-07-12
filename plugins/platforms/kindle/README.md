# Hermes Notebook platform

The Notebook platform connects handwriting-oriented clients to the normal
Hermes Gateway message pipeline:

```text
Kindle / iPad / Android / BOOX -> notebook bridge -> authenticated localhost ingest -> Hermes Gateway
```

This adapter deliberately does not expose Hermes to the LAN or internet. It
listens on `127.0.0.1`, authenticates the companion bridge with
`KINDLE_INGEST_TOKEN`, applies the configured Hermes user allowlist, and turns
each accepted note into a standard `MessageEvent`. Models, memory, sessions,
skills, and `platform_toolsets.kindle` then work exactly as they do for other
messaging platforms.

The registered platform name remains `kindle` for configuration and session
compatibility. Existing Kindle bridges continue to work unchanged.

## Supported client families

- Kindle Scribe browser clients.
- iPadOS clients using Apple Pencil.
- Android tablet clients using a stylus, including Samsung S Pen devices.
- BOOX Android/e-ink tablets using pen input.

New clients should send `X-Notebook-Token`; `X-Kindle-Token` remains supported.
Set `NOTEBOOK_INGEST_TOKEN` for a shared notebook secret, or keep the existing
`KINDLE_INGEST_TOKEN` variable during migration.

## Companion bridge

This platform expects a companion bridge process to own the Kindle-specific UI,
OCR, and remote-browser boundary. The bridge POSTs authenticated notebook turns
to this adapter's localhost `/ingest` endpoint. A bridge can support:

- Full-page Kindle Scribe pen input.
- Vision OCR followed by an optional text-model proper-name cleanup pass.
- Stable Hermes thread IDs for new and continued notebook entries.
- Artifact/image/HTML annotation workspaces.
- Notebook-consistent artifact controls: HTML/artifact surfaces should use the
  same compact, icon-first menu language as the writing surface, with matching
  quick intents such as summarize, tasks, creative draft, email draft, and
  workpaper note.
- Creative notebook turns for drafting, remixing, worldbuilding, brainstorming,
  and interpretation that should not be forced into a business/workpaper frame.
- Passwordless trusted-LAN operation or explicit browser authentication.
- Away-from-LAN access through Tailscale Funnel with a permanent, high-entropy
  `/remote/<key>` bookmark path that does not depend on Kindle cookies, query
  parameters, or local storage.

## Gateway configuration

Set a random bridge-to-adapter secret and restrict the channel to known users:

```powershell
$env:KINDLE_INGEST_TOKEN = '<random-secret>'
$env:KINDLE_ALLOWED_USERS = 'kindle-user'
```

Configure the tools this platform may use. The Kindle channel is meant to reach
the real Hermes agent, so do not leave it as a plain/no-tools notebook unless
you are intentionally sandboxing it. Give `platform_toolsets.kindle` the same
tools the Kindle user should be able to ask Hermes to use from the Scribe:

```yaml
platform_toolsets:
  kindle:
    - memory
    - file
    - terminal
    - web
    - live-page
    - vision
```

Then start the Gateway and verify `http://127.0.0.1:8793/health` before starting
the companion bridge.

The companion bridge should route normal Kindle sends through this adapter. It
should not default to a separate plain/local model path, because that makes the
same notebook randomly lose tools and memory depending on a UI toggle or stale
browser storage.

For MoA or tool-heavy notebook turns, raise the synchronous reply window instead
of letting the Kindle surface fail early:

```powershell
$env:KINDLE_REPLY_TIMEOUT = '360'
```

## Companion payload hints

The ingest body must include `text`, `user`, and `chat_id`. The companion bridge
may also send optional metadata that this adapter converts into agent-visible
context:

- `client`: device metadata such as
  `{"platform":"ipados","stylus":"apple-pencil","capabilities":["pressure","tilt"]}`.
  Supported platform values are `kindle`, `ipados`, `android`, and `boox`.
- `device_platform`, `stylus`, and `capabilities`: flat compatibility fields for
  simpler clients.

- `intent`: one of `summarize`, `tasks`, `creative`, `email`, or `workpaper`.
- `tags`: an array or comma-separated list such as `["client", "todo"]`.
- `source` / `artifact_type`: use `source: "live-page"` or
  `artifact_type: "html"` when the Kindle user is annotating rendered HTML.
- `ocr_raw` and `ocr_cleaned`: raw and cleaned handwriting transcriptions when
  ambiguity matters.

These hints support notebook QoL without making the adapter own UI behavior:
the bridge can show one-tap intents, progress heartbeats, and short-first
responses while the agent still performs tool work. When the bridge marks a note
as live-page or HTML work, the adapter reminds the agent to update the configured
live-page/artifact display so newly generated HTML replaces the old visible
page.

The `workpaper` intent is only one mode. For open-ended sketches, fiction,
poetry, product concepts, personal notes, or "what does this make you think of?"
turns, the bridge should send `creative` or omit `intent` entirely so the agent
can interpret the note without narrowing it to Bearden or business workflows.

## Remote-access security boundary

Do not expose port `8793`. Remote browser access terminates at the companion
bridge; the bridge remains the only caller of this adapter. If using Tailscale
Funnel, configure `DIARY_REMOTE_KEY` in the companion bridge before enabling the
public HTTPS route. Remote requests without that key must fail before they can
reach this adapter, session history, or stored handwriting.
