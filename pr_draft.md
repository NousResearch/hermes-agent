# feat(gateway): add Matrix protocol platform adapter

## Summary

Full Matrix platform adapter for the Hermes gateway. Agents can be deployed as always-on Matrix bots with E2EE, cross-signing, media support, typing indicators, and a zero-friction setup wizard that handles the entire configuration in one pass.

## What changed

**21 files, ~3,800 lines**

| File | Change |
|------|--------|
| `gateway/platforms/matrix.py` | New adapter (~1,320 lines) |
| `hermes_cli/gateway.py` | Setup wizard + `verify-matrix` command (+1,000 lines) |
| `tests/gateway/test_matrix.py` | 93 tests, all passing |
| `website/docs/user-guide/messaging/matrix.md` | Full setup guide (new) |
| `hermes_cli/config.py` | 10 new Matrix env vars in `OPTIONAL_ENV_VARS` |
| `gateway/config.py` | `Platform.MATRIX` enum + env var loading |
| `gateway/run.py` | Adapter factory + authorization maps |
| `hermes_cli/main.py` | `hermes gateway verify-matrix` subcommand |
| `toolsets.py` | `hermes-matrix` toolset |
| `tools/send_message_tool.py` | `_send_matrix()` standalone send function |
| `tools/cronjob_tools.py`, `cron/scheduler.py` | Matrix delivery option |
| `gateway/channel_directory.py` | Session-based discovery |
| `agent/prompt_builder.py` | `PLATFORM_HINTS["matrix"]` |
| `agent/redact.py` | `_MATRIX_ID_RE` pattern |
| `hermes_cli/status.py`, `hermes_cli/setup.py` | Matrix in status/main wizard |
| `website/docs/user-guide/messaging/index.md` | Architecture diagram, toolset table |

## ADDING_A_PLATFORM.md checklist

All 16 items verified:

| # | Item | Status |
|---|------|--------|
| 1 | Core adapter — all required + optional methods, `check_matrix_requirements()` | ✅ |
| 2 | Platform enum + env var loading (`gateway/config.py`) | ✅ |
| 3 | Adapter factory (`gateway/run.py`) | ✅ |
| 4 | Authorization maps — both `platform_env_map` and `platform_allow_all_map` | ✅ |
| 5 | Session source — `build_source()` used (no custom fields needed) | ✅ |
| 6 | System prompt hints (`agent/prompt_builder.py`) | ✅ |
| 7 | Toolset — `hermes-matrix` in `toolsets.py` + added to `hermes-gateway` composite | ✅ |
| 8 | Cron delivery (`cron/scheduler.py` platform map) | ✅ |
| 9 | Send message tool — `_send_matrix()` + platform routing in `send_message_tool.py` | ✅ |
| 10 | Cronjob tool schema — `deliver` param updated | ✅ |
| 11 | Channel directory — `"matrix"` in session-based discovery list | ✅ |
| 12 | Status display (`hermes_cli/status.py`) | ✅ |
| 13 | Gateway setup wizard — full `_setup_matrix()` with auto-login and E2EE flow | ✅ |
| 14 | ID redaction — `_MATRIX_ID_RE` in `agent/redact.py` | ✅ |
| 15 | Documentation — `matrix.md` (new), `index.md`, `security.md`, `README.md` | ✅ |
| 16 | Tests — 93 tests in `tests/gateway/test_matrix.py` | ✅ |

## Technical design

### Library: mautrix-python (not matrix-nio)

matrix-nio was evaluated and found unsuitable: cross-signing unimplemented since 2020 (issue #229), unreliable session persistence, last maintained 2023. mautrix-python is used by maubot and all production mautrix bridges.

### E2EE

Optional (`MATRIX_E2EE=true`). When enabled:

- **`PgCryptoStore` on SQLite** (`~/.hermes/matrix/crypto.db`) — all Olm/Megolm sessions survive restarts, eliminating "no session found" decrypt errors. Same approach as maubot.
- **`resolve_trust()` guard** — cross-signing bootstrap only runs once. Subsequent restarts check the local DB and skip if already verified.
- **Recovery key auto-saved** to `~/.hermes/.env` on first bootstrap so device self-signing works on every restart.
- **Ghost device purge** on startup — Synapse creates phantom device entries from `keys/device_signing/upload` calls. These cause Element to encrypt Olm messages to non-existent accounts. The adapter detects and removes them with UIA fallback.

### Python 3.14 compatibility

mautrix 0.21.0 has five incompatibilities with Python 3.14, addressed via `_patch_mautrix_py314()` applied at import time. No-op on Python < 3.14. Two of the five patches fix real mautrix bugs present on all versions.

### Setup wizard

`hermes gateway setup matrix` handles the entire setup in one pass:
1. Homeserver URL + SSL + connectivity test
2. Bot user ID
3. Bot password → wizard logs in, gets token + device ID automatically
4. E2EE deps checked and installed if needed
5. Cross-signing bootstrap with recovery key auto-saved
6. Allowed users
7. Trust verification — wizard signs bot's master key from each allowed user's account

No curl commands, no manual token copying, no separate steps.

### Media support

Full inbound and outbound media:
- **Outbound**: `send_image`, `send_image_file`, `send_animation`, `send_voice`, `send_video`, `send_document` — all upload to Matrix media repository via `/_matrix/media/v3/upload` then send as typed message events (`m.image`, `m.audio`, `m.video`, `m.file`). Accepts local file paths or existing `mxc://` URIs.
- **Inbound**: Downloads via `/_matrix/media/v3/download`, caches to `~/.hermes/cache/`, populates `event.media_urls` and `event.media_types` following the same pattern as the Telegram adapter so vision tools and STT transcription work correctly.

### Typing indicators

`send_typing()` sends `PUT /rooms/{id}/typing` and starts a 20s keepalive loop (Matrix typing expires at 30s). Cleared automatically when the response is sent. Same pattern as nanobot's Matrix implementation.

### Other checklist items
- `MAX_MESSAGE_LENGTH = 65535` (Matrix has no spec limit; homeserver default)
- `build_source()` used for `SessionSource` construction
- All send_* methods use correct parameter names matching base class signatures

## Known limitations

1. **`_send_matrix()` in `send_message_tool.py` does not support E2EE** — uses a one-shot HTTP client, no sync loop. Messages to encrypted rooms will fail silently. Documented with a warning log when `MATRIX_E2EE=true`. Requires a running gateway for E2EE rooms.

2. **SAS verification ceremony** — implemented using `olm.Sas` directly (mautrix 0.21.0 has no built-in SAS support). Tested with Element Desktop; other clients' verification flows may need adjustment.

3. **No sync reconnection backoff** — mautrix's `client.start()` has internal retry logic but no explicit exponential backoff wrapper. Added to future work.

4. **`MATRIX_RECOVERY_KEY` rotation** — if the crypto DB is wiped, the stored recovery key becomes stale. The setup wizard's "Wipe E2EE state" option clears it; re-running setup re-establishes everything.

## Testing

### Tested against
- **Synapse** self-hosted (k3s, v1.99+), self-signed TLS via Tailscale
- **Element Desktop** on Arch Linux
- **Python 3.14** (primary development environment)
- **Media upload/download** — images, audio, documents all tested against the live Synapse instance

### k8s / self-hosted Synapse note
If you're running Synapse on k8s and media upload returns `500 Internal server error`, check that the `media_store` directory is writable by the Synapse process user (uid 991). This can happen when the volume is provisioned by root. Fix:
```bash
kubectl exec -n matrix <synapse-pod> -- chown -R 991:991 /data/media_store
```
The init container in the standard Synapse k8s deployment should handle this on every restart via `chown -R 991:991 /data`, but existing volumes may need a one-time manual fix.

### Needs community validation
- **matrix.org** (public homeserver) — rate limits and UIA behavior may differ
- **Dendrite** — different cross-signing implementation
- **Element X** (Rust SDK) — uses a different verification flow
- **Python 3.11 / 3.12** — patches are no-ops; should work but needs confirmation
- **Other clients** (FluffyChat, Nheko, Cinny)

### Test results
```
93 passed (gateway/test_matrix.py)
3550 passed total — same 35 pre-existing failures on main, zero regressions
```

### How to test

**Prerequisites:** Synapse (or compatible), two Matrix accounts, Python deps:
```bash
pip install "mautrix[e2be]" asyncpg aiosqlite base58
sudo pacman -S libolm  # or apt install libolm-dev
```

**Basic flow:**
```bash
hermes gateway setup matrix
# Enter homeserver URL, bot password (wizard does the rest)

hermes gateway restart
# Send a message in Element — bot responds within seconds
```

**Media:**
```bash
# Send an image/audio/file to the bot in Element
# It should be processed by the vision/STT pipeline
```

**Re-run trust if needed:**
```bash
hermes gateway verify-matrix
```

## Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `MATRIX_HOMESERVER_URL` | Yes | e.g. `https://matrix.example.org` |
| `MATRIX_ACCESS_TOKEN` | Yes | Bot account access token (`syt_...`) |
| `MATRIX_USER_ID` | Yes | Bot Matrix ID (`@bot:example.org`) |
| `MATRIX_DEVICE_ID` | Recommended | Prevents session replay on restart |
| `MATRIX_ALLOWED_USERS` | Recommended | Comma-separated IDs allowed to message the bot |
| `MATRIX_HOME_CHANNEL` | Optional | Room ID for cron job delivery |
| `MATRIX_HOME_CHANNEL_NAME` | Optional | Display name for home channel (default: "Home") |
| `MATRIX_VERIFY_SSL` | Optional | `false` for self-signed TLS |
| `MATRIX_E2EE` | Optional | `true` for E2EE (requires deps above) |
| `MATRIX_PASSWORD` | Optional | Bot password for cross-signing UIA |
| `MATRIX_RECOVERY_KEY` | Optional | Auto-saved; used for device self-signing on restart |

## Commit history

```
feat(gateway): add Matrix protocol platform adapter
adding appropriate documentation and better wizard setup for matrix gateway
adding appropriate documentation and better wizard setup for matrix gateway
proper setup wizard using mautrix, better restart policy, it works now
```

---

*Tested on: Arch Linux, Python 3.14, Synapse self-hosted, Element Desktop*
*Branch: feat/matrix-gateway*
