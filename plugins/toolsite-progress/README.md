# Toolsite Progress Plugin

Local Toolsite remote-control plugin for Hermes gateway.

## Runtime Path

The active runtime plugin for this machine is:

```txt
/Users/dom/agents/hermes-toolsite-monitor/hermes-home/plugins/toolsite-progress/
```

This tracked source copy lives at:

```txt
plugins/toolsite-progress/
```

## Sync To Runtime

From the Hermes repository root:

```bash
mkdir -p /Users/dom/agents/hermes-toolsite-monitor/hermes-home/plugins/toolsite-progress
cp plugins/toolsite-progress/__init__.py /Users/dom/agents/hermes-toolsite-monitor/hermes-home/plugins/toolsite-progress/__init__.py
cp plugins/toolsite-progress/plugin.yaml /Users/dom/agents/hermes-toolsite-monitor/hermes-home/plugins/toolsite-progress/plugin.yaml
python3 -m py_compile /Users/dom/agents/hermes-toolsite-monitor/hermes-home/plugins/toolsite-progress/__init__.py
```

Restart Hermes gateway after syncing so the hook reloads.

## Telegram Image Attachments

Telegram's adapter caches incoming photos as local `MessageEvent.media_urls`.
This plugin copies image media into:

```txt
${HERMES_HOME}/state/toolsite-attachments/<chat_id>/<message_id>/
```

It appends the normal Hermes inbox JSONL record to:

```txt
${HERMES_HOME}/state/toolsite-inbox.jsonl
```

Image records include:

```json
{
  "attachments": [
    {
      "kind": "image",
      "telegram_file_id": "...",
      "local_path": "...",
      "mime_type": "image/jpeg",
      "file_name": "image.jpg",
      "width": 1600,
      "height": 1200
    }
  ]
}
```

Caption text is preserved as `text`. Plain text messages continue to write the
same inbox shape as before and do not include `attachments`.

## Configuration

Optional environment overrides:

- `HERMES_HOME`
- `TOOL_SITE_REMOTE_STATE_DIR`
- `TOOL_FACTORY_REPO`
- `TOOL_SITE_STATUS_SCRIPT`
- `TOOL_SITE_ADVISOR_SCRIPT`

## Verification

```bash
python3 -m py_compile plugins/toolsite-progress/__init__.py
uv run --extra dev pytest -q tests/plugins/test_toolsite_progress_plugin.py
```
