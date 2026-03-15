# feat(gateway): add Matrix protocol platform adapter

## Summary

Full Matrix platform adapter for the Hermes gateway. Agents can be deployed as always-on Matrix bots with optional E2EE, cross-signing verification, full media support (inbound and outbound), typing indicators, and a zero-friction setup wizard that handles the entire configuration automatically in one pass.

---

## Files changed

**23 files, ~4,650 lines**

| File | Change |
|------|--------|
| `gateway/platforms/matrix.py` | New adapter (~1,480 lines) |
| `hermes_cli/gateway.py` | Setup wizard + `verify-matrix` command (+1,000 lines) |
| `tests/gateway/test_matrix.py` | 93 tests, all passing |
| `website/docs/user-guide/messaging/matrix.md` | Full setup guide (new, 270 lines) |
| `hermes_cli/config.py` | 10 new Matrix env vars in `OPTIONAL_ENV_VARS` |
| `gateway/config.py` | `Platform.MATRIX` enum + env var loading |
| `gateway/run.py` | Adapter factory, auth maps, stale library name fix |
| `hermes_cli/main.py` | `hermes gateway verify-matrix` subcommand (+109 lines) |
| `toolsets.py` | `hermes-matrix` toolset + added to `hermes-gateway` |
| `tools/send_message_tool.py` | `_send_matrix()` standalone send function |
| `tools/cronjob_tools.py`, `cron/scheduler.py` | Matrix delivery option |
| `gateway/channel_directory.py` | Session-based discovery |
| `agent/prompt_builder.py` | `PLATFORM_HINTS["matrix"]` |
| `agent/redact.py` | `_MATRIX_ID_RE` pattern |
| `hermes_cli/status.py`, `hermes_cli/setup.py` | Matrix in status/main wizard |
| `website/docs/user-guide/messaging/index.md` | Architecture diagram, toolset table |
| `website/docs/user-guide/security.md` | `MATRIX_ALLOWED_USERS` example |
| `tools/terminal_tool.py` | Fix stale `minisweagent` import (pre-existing bug caught during review) |

---

## ADDING_A_PLATFORM.md checklist — all 16 items

| # | Item | Status |
|---|------|--------|
| 1 | Core adapter — all required + optional methods, correct base signatures, `check_matrix_requirements()` | ✅ |
| 2 | Platform enum + env var loading (`gateway/config.py`) | ✅ |
| 3 | Adapter factory (`gateway/run.py`) | ✅ |
| 4 | Authorization maps — `platform_env_map` and `platform_allow_all_map` | ✅ |
| 5 | Session source — `build_source()` used (no new fields needed) | ✅ |
| 6 | System prompt hints (`agent/prompt_builder.py`) | ✅ |
| 7 | Toolset — `hermes-matrix` + added to `hermes-gateway` composite | ✅ |
| 8 | Cron delivery (`cron/scheduler.py`) | ✅ |
| 9 | Send message tool — `_send_matrix()` + platform routing | ✅ |
| 10 | Cronjob tool schema — `deliver` param | ✅ |
| 11 | Channel directory — `"matrix"` in session-based discovery | ✅ |
| 12 | Status display (`hermes_cli/status.py`) | ✅ |
| 13 | Gateway setup wizard — full `_setup_matrix()` with auto-login, E2EE, trust verification | ✅ |
| 14 | ID redaction — `_MATRIX_ID_RE` in `agent/redact.py` | ✅ |
| 15 | Docs — `matrix.md` (new), `index.md`, `security.md`, `README.md` | ✅ |
| 16 | Tests — 93 tests in `tests/gateway/test_matrix.py` | ✅ |

---

## Technical design

### Library: mautrix-python (not matrix-nio)

matrix-nio was evaluated and found unsuitable for production E2EE bots:
- Cross-signing unimplemented since 2020 (issue #229, open, no activity)
- No reliable session persistence — every restart loses all Olm/Megolm sessions, causing "no session found" decrypt errors on every message after a restart
- Last maintained 2023

mautrix-python is the library used by maubot and all production mautrix bridges. It has `PgCryptoStore` for proper persistence and active maintenance.

### E2EE design (mirrors maubot)

- **`PgCryptoStore` on SQLite** (`~/.hermes/matrix/crypto.db`) — all Olm/Megolm sessions, device keys, and cross-signing keys survive gateway restarts. This is the fundamental fix vs matrix-nio.
- **`resolve_trust()` guard** — cross-signing bootstrap runs once on first start only. All subsequent restarts skip it instantly via the local DB.
- **Recovery key auto-saved** to `~/.hermes/.env` on first bootstrap so device self-signing works on every restart.
- **Ghost device purge** — Synapse stores cross-signing public keys as device table rows. Without removal, Element tries to encrypt Olm messages to these phantom accounts and decryption fails. The adapter removes them on startup with UIA password fallback.
- **Encrypted media** — inbound files in E2EE rooms use `content.file.url` (not `content.url`) and require `decrypt_attachment` before caching. Both handled.

### Python 3.14 compatibility

mautrix 0.21.0 has five incompatibilities with Python 3.14. `_patch_mautrix_py314()` applies targeted monkey-patches at import time, no-op on Python < 3.14. Two patches fix real mautrix bugs on all Python versions and run unconditionally.

### Media support

**Outbound** — all methods use correct base-class parameter names and upload via `/_matrix/media/v3/upload`. Supports `send_image`, `send_image_file`, `send_animation`, `send_voice`, `send_video`, `send_document`. Accepts local paths or existing `mxc://` URIs.

**Inbound** — mautrix msgtype strings (`"m.image"`, `"m.audio"`, `"m.video"`, `"m.file"`) compared as strings. Downloads, decrypts if E2EE, caches, and populates `event.media_urls`/`event.media_types` following the Telegram adapter pattern exactly. Agent sees files via the standard document context note.

### Typing indicators

`send_typing()` → `PUT /rooms/{id}/typing` with 20s keepalive (Matrix typing expires at 30s). Clears in `send_message()`. Same pattern as nanobot's Matrix implementation.

### Setup wizard (zero manual steps)

`hermes gateway setup matrix`:
1. Homeserver URL + SSL + live connectivity test
2. Bot user ID
3. Bot password → logs in automatically, gets token + device ID (no curl commands)
4. E2EE deps checked/installed
5. Cross-signing bootstrap inline, recovery key auto-saved
6. Allowed users
7. Trust verification inline — signs bot's master key from each allowed user's account

Re-configuration: wizard offers to wipe all E2EE state for a clean start.

---

## Known limitations

1. **`_send_matrix()` tool does not support E2EE** — one-shot HTTP client, no sync loop. Messages to encrypted rooms will fail. Documented with warning log. The running gateway handles E2EE correctly; this only affects the standalone tool.

2. **SAS verification** — implemented via `olm.Sas` directly (mautrix 0.21.0 lacks built-in SAS). Tested with Element Desktop only.

3. **No explicit sync reconnection backoff** — mautrix's `client.start()` has internal retry but no configurable backoff wrapper.

---

## Testing

### Tested against
- **Synapse** self-hosted on k3s (v1.99+), self-signed TLS via Tailscale
- **Element Desktop** on Arch Linux
- **Python 3.14**
- **Media** — image, audio, document upload/download verified live against Synapse

### k8s / self-hosted note
If media upload returns `500 Internal server error`, the `media_store` volume may be owned by root. Fix:
```bash
kubectl exec -n matrix <pod> -- chown -R 991:991 /data/media_store
```
The standard Synapse k8s init container handles this on every pod start, but pre-existing volumes may need a one-time manual fix.

### Needs community validation
- **matrix.org** (public homeserver) — rate limits and UIA behavior may differ
- **Dendrite** — different cross-signing implementation
- **Element X** (Rust SDK) — different verification flow
- **Python 3.11 / 3.12** — patches are no-ops; should work but unconfirmed
- **Other clients** (FluffyChat, Nheko, Cinny) — untested

### How to test

```bash
# Install E2EE dependencies
# Setup wizard does handle this and install if not available
pip install "mautrix[e2be]" asyncpg aiosqlite base58
sudo pacman -S libolm   # Arch; or: apt install libolm-dev

# Full guided setup (wizard handles everything)
hermes gateway setup
# select matrix (6)

# Start
hermes gateway restart

# Send a text message, image, PDF in Element — bot should respond
# If trust verification incomplete:
hermes gateway verify-matrix
```

### Test results
```
93 passed  (tests/gateway/test_matrix.py)
3550 passed total — same 35 pre-existing failures as main, zero regressions
```

---

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MATRIX_HOMESERVER_URL` | Yes | e.g. `https://matrix.example.org` |
| `MATRIX_ACCESS_TOKEN` | Yes | Bot account access token (`syt_...`) |
| `MATRIX_USER_ID` | Yes | Bot Matrix ID (`@bot:example.org`) |
| `MATRIX_DEVICE_ID` | Recommended | Pins to one device; prevents session replay on restart |
| `MATRIX_ALLOWED_USERS` | Recommended | Comma-separated IDs allowed to message the bot |
| `MATRIX_HOME_CHANNEL` | Optional | Room ID for cron job delivery |
| `MATRIX_HOME_CHANNEL_NAME` | Optional | Display name for home channel (default: "Home") |
| `MATRIX_VERIFY_SSL` | Optional | `false` for self-signed TLS (default `true`) |
| `MATRIX_E2EE` | Optional | `true` for E2EE (requires deps above) |
| `MATRIX_PASSWORD` | Optional | Bot password for cross-signing UIA fallback |
| `MATRIX_RECOVERY_KEY` | Optional | Auto-saved by gateway; enables device self-signing on restart |

## Dependencies

All optional — only needed when `MATRIX_E2EE=true`. Installed automatically by the setup wizard.

| Package | Purpose |
|---------|---------|
| `mautrix[e2be]` | Matrix client + E2EE crypto |
| `asyncpg` | SQL dialect for `PgCryptoStore` |
| `aiosqlite` | SQLite backend (no PostgreSQL needed) |
| `base58` | Recovery key encoding |
| `libolm` | Olm/Megolm C library (system package) |

---

*Branch: `feat/matrix-gateway` | Python 3.14 | Synapse self-hosted | Element Desktop*
