# Hypura Native Integration

Hermes now exposes a native command surface for the migrated Hypura/VRChat/Ollama stack.

## Mirrored assets

- `vendor/openclaw-mirror/` (full mirror of OpenClaw `vendor/`)
- `vendor/openclaw-mirror/extensions/hypura-harness`
- `vendor/openclaw-mirror/extensions/hypura-provider`
- `vendor/openclaw-mirror/extensions/vrchat-relay`

## CLI commands

```bash
hermes hypura status
hermes hypura start-daemon
hermes hypura osc --action chatbox --payload '{"text":"hello"}'
hermes hypura speak --text "hello world" --emotion joy
hermes hypura run --task "analyze this dataset"
hermes hypura scientist-run --topic "VR interaction" --num-ideas 3
hermes hypura vrchat-chatbox --message "ASI_ACCEL"
hermes hypura ollama-start --pull-model qwen2.5:7b
hermes hypura ollama-status
```

## Environment variables

- `HYPURA_HARNESS_URL`
- `HYPURA_HARNESS_SCRIPT`
- `HYPURA_API_KEY`
- `OLLAMA_API_KEY` (optional local synthetic key)
- `OLLAMA_BASE_URL`

## Notes

- The Hypura daemon is expected at `vendor/openclaw-mirror/extensions/hypura-harness/scripts/harness_daemon.py` unless overridden.
- `hermes hypura vrchat-chatbox` uses Hypura `/osc` action `chatbox`.
- This integration is native Hermes CLI orchestration and does not require OpenClaw runtime.
- For the safer Neuro-sama-style autonomy path, see `docs/migration/vrchat_neurosama_autonomy.md`.
