# File Uploads

Hermes Agent can attach files to your session — images, PDFs, source code,
configs, CSVs, and more — from the TUI composer, drag-and-drop, or the
`/attach` slash command.

## How to attach

You can attach a file in three ways:

1. **Drag and drop a file onto the terminal.** Most modern terminals
   (Kitty, WezTerm, iTerm2, foot) pass the file path as a paste event.
   Hermes detects paths automatically.

2. **Paste a path into the composer.** If the text you paste looks like
   a file path (`/abs/path`, `~/foo`, `./bar`, `C:\Users\…`), Hermes
   detects it and attaches the file on submit.

3. **Use the `/attach` command.** Type `/attach <path>` (alias `/file`).
   TAB completion is built into the composer — start typing any path-like
   prefix (`/`, `~/`, `./`, `../`) and press TAB to see filesystem
   candidates.

The legacy `/image` command is preserved as an alias and now also routes
through the same pipeline.

## What you can attach

By default, the following MIME types are accepted (detected by file
content, not extension, via libmagic):

- **Images**: PNG, JPEG, GIF, WebP, BMP, TIFF, SVG, ICO
- **Documents**: PDF
- **Text / code**: plain text, Markdown, CSV, HTML, XML, YAML
- **Structured data**: JSON, YAML, TOML
- **Data**: Parquet

To extend the whitelist, set `uploads.allowed_mime_types` in your
`~/.hermes/config.yaml`. To disable the whitelist entirely, set
`uploads.allowed_mime_types: ["*"]` (use with care).

## Limits

| Limit | Default | Config key |
|---|---|---|
| Per-file size | 10 MB | `uploads.max_size_mb` |
| Per-session aggregate | 50 MB | `uploads.max_session_mb` |
| Allowed MIME types | see list above | `uploads.allowed_mime_types` (use `["*"]` to disable) |

Files over the per-file limit are rejected with a clear error message
and never copied to the sandbox.

## Inline previews (images)

If your terminal supports one of the standard image protocols, attached
images are rendered as inline previews in the chat. Supported terminals:

| Terminal | Protocol | Quality |
|---|---|---|
| Kitty | Kitty Graphics Protocol | Best; animated GIFs |
| iTerm2 (macOS) | OSC 1337 | Best |
| WezTerm, foot, mlterm | Sixel | Good; no animation |
| Plain xterm, Terminal.app | (none) | Text-only metadata |

To check, attach a PNG and watch the chat — if you see a tiny rendered
image, it works. Otherwise you'll see the text-only metadata line
(`📎 Attached image: foo.png · 2.0 KB · image/png`).

## Sandboxing

Attached files are copied to `/tmp/hermes-uploads/<session_id>/` with
`chmod 600` (owner read/write only). The original file is never moved
or modified. When your session ends, the sandbox directory is removed.

## Spoof detection

Hermes detects MIME type by **file content** (using libmagic), not by
extension. A `malware.exe` renamed to `foto.png` is rejected as
`application/x-msdownload`, not accepted as `image/png`. This is the
single most important security property of the upload system — never
trust the user-supplied filename or extension.

## JSON-RPC API (for tool builders)

| Method | Params | Returns |
|---|---|---|
| `file.attach` | `{session_id, path}` | `{attached, id, name, stored_path, mime_type, size_bytes, kind, preview_text, remainder, width?, height?, token_estimate?}` |
| `file.list` | `{session_id}` | `{files: [...]}` |
| `file.detach` | `{session_id, id}` | `{detached, id}` |

Error codes: `4015` (path required), `4016` (not found), `4017`
(whitelist/size rejection), `5028-5030` (server error).
