feat(gateway): add Matrix protocol platform adapter
Summary
This PR adds a full Matrix platform adapter to the Hermes gateway, enabling the agent to operate as an always-on Matrix bot. It covers all 16 integration points from gateway/platforms/ADDING_A_PLATFORM.md and includes a zero-friction setup wizard that handles the entire setup flow — including E2EE, cross-signing, and trust verification — automatically.
---
What's included
21 files changed, ~3,800 lines
| File | Change |
|------|--------|
| gateway/platforms/matrix.py | New adapter (~1,260 lines) |
| hermes_cli/gateway.py | Setup wizard + verify-matrix command (+1,000 lines) |
| tests/gateway/test_matrix.py | 76 tests covering all public API surfaces |
| website/docs/user-guide/messaging/matrix.md | Full setup guide (new) |
| hermes_cli/config.py | 9 new Matrix env vars in OPTIONAL_ENV_VARS |
| gateway/config.py | Platform.MATRIX enum + env var loading |
| gateway/run.py | Adapter factory + authorization maps |
| hermes_cli/main.py | hermes gateway verify-matrix subcommand |
| toolsets.py | hermes-matrix toolset |
| tools/send_message_tool.py | _send_matrix() standalone send function |
| tools/cronjob_tools.py, cron/scheduler.py | Matrix delivery option |
| gateway/channel_directory.py | Session-based discovery |
| agent/prompt_builder.py | PLATFORM_HINTS["matrix"] |
| agent/redact.py | _MATRIX_ID_RE pattern + redaction |
| hermes_cli/status.py, hermes_cli/setup.py | Matrix in status/main wizard |
| website/docs/user-guide/messaging/index.md | Architecture diagram, toolset table |
---
Technical design
Library choice: mautrix-python over matrix-nio
The implementation uses mautrix-python (https://github.com/mautrix/python) rather than matrix-nio. This was a deliberate choice after matrix-nio was evaluated and found to lack:
- Cross-signing support (issue #229, open since December 2020, no activity)
- Reliable Olm/Megolm session persistence across restarts
- Active maintenance (last commit 2023)
mautrix-python is the library used by maubot and all production mautrix bridges. It has proper E2EE session persistence via PgCryptoStore, built-in key management, and active maintenance.
E2EE design
E2EE is optional but fully supported when MATRIX_E2EE=true. The design mirrors maubot's approach:
- PgCryptoStore on SQLite (~/.hermes/matrix/crypto.db) — all Olm sessions, inbound Megolm sessions, device keys, and cross-signing keys survive gateway restarts. This eliminates the "no session found" decrypt errors that would otherwise occur on every restart.
- resolve_trust() guard — cross-signing bootstrap (generate_recovery_key) only runs once. On all subsequent restarts, resolve_trust() finds the stored signatures in the DB and returns immediately — no re-upload, no UIA challenge, no network round-trips.
- Recovery key auto-saved — the gateway writes MATRIX_RECOVERY_KEY to ~/.hermes/.env on first bootstrap so device self-signing works on every restart without user intervention.
- Ghost device purge — Synapse creates phantom device entries when keys/device_signing/upload is called. These cause Element to try encrypting Olm messages to non-existent Olm accounts, breaking decryption. The adapter detects and removes these on every startup with UIA fallback.
Python 3.14 compatibility
mautrix 0.21.0 has five incompatibilities with Python 3.14, addressed via _patch_mautrix_py314() in matrix.py. This function is a no-op on Python < 3.14 and applies targeted monkey-patches at import time. Two of the five patches fix real mautrix bugs and now run on all Python versions (not just 3.14).
Setup wizard: zero-friction first-time setup
The hermes gateway setup matrix wizard handles the entire setup in one pass:
1. Enter homeserver URL → live connectivity test
2. Enter bot account password → wizard logs in, gets token + device ID automatically
3. E2EE deps checked, libolm + mautrixe2be installed if needed
4. Cross-signing bootstrap runs inline — recovery key auto-saved to .env
5. Enter your Matrix user ID as allowed user
6. Trust verification: wizard logs in as the allowed user and signs the bot's master key
No curl commands, no manual token copying, no separate verify-matrix step for standard setups.
---
ADDING_A_PLATFORM.md checklist
All 16 items from gateway/platforms/ADDING_A_PLATFORM.md:
| # | Item | Status |
|---|------|--------|
| 1 | Core adapter (gateway/platforms/matrix.py) | ✅ |
| 2 | Platform enum + env var loading (gateway/config.py) | ✅ |
| 3 | Adapter factory (gateway/run.py) | ✅ |
| 4 | Authorization maps — both platform_env_map and platform_allow_all_map | ✅ |
| 5 | Session source — no new fields needed; Matrix uses existing chat_id/user_id | ✅ N/A |
| 6 | System prompt hints (agent/prompt_builder.py) | ✅ |
| 7 | Toolset — hermes-matrix in toolsets.py + added to hermes-gateway | ✅ |
| 8 | Cron delivery (cron/scheduler.py platform map) | ✅ |
| 9 | Send message tool — _send_matrix() + platform routing | ✅ |
| 10 | Cronjob tool schema — deliver param updated | ✅ |
| 11 | Channel directory — "matrix" in session-based discovery list | ✅ |
| 12 | Status display (hermes_cli/status.py) | ✅ |
| 13 | Gateway setup wizard — full _setup_matrix() with custom E2EE flow | ✅ |
| 14 | Phone/ID redaction — _MATRIX_ID_RE in agent/redact.py | ✅ |
| 15 | Documentation — matrix.md (new), index.md, security.md, README.md | ✅ |
| 16 | Tests — 76 tests in tests/gateway/test_matrix.py | ✅ |
---
Known limitations and testing notes
Tested against
- Synapse (self-hosted, v1.99+) via Tailscale/private network with self-signed TLS
- Element Desktop on Arch Linux
- Python 3.14 (the development environment for this PR)
Needs validation from community
- matrix.org (public homeserver) — rate limits and key upload behavior may differ
- Dendrite — different cross-signing implementation, may need UIA handling adjustments
- Element X (Rust SDK) — uses a different verification flow; SAS handlers are implemented but untested against Element X specifically
- Python 3.11 / 3.12 — the _patch_mautrix_py314() guard means these users get the unpatched mautrix. Works on 3.14; should also work on 3.11/3.12 but needs confirmation
- Other clients (FluffyChat, Nheko, Cinny) — untested; the adapter uses standard CS API calls but client behavior varies
Known limitations (documented)
1. _send_matrix() in send_message_tool.py does not support E2EE. It uses a one-shot HTTP client with no sync loop, so it cannot participate in Megolm sessions. Messages sent via the tool to encrypted rooms will fail silently on the client side. This is documented in the code with a warning log when MATRIX_E2EE=true. Resolving this requires a more complex approach (running a minimal sync loop) which is out of scope for this PR.
2. send_typing() not implemented — the base class provides a no-op stub. Matrix supports typing notifications (PUT /rooms/{id}/typing) but this is low-priority.
3. send_voice() / send_video() / send_animation() — base class stubs used; these would fall back to send_document().
4. SAS verification ceremony — the SAS handlers are implemented using olm.Sas directly (mautrix 0.21.0 has no built-in SAS support). Tested with Element Desktop; other clients' verification flows may need adjustment.
5. MATRIX_RECOVERY_KEY not rotated automatically — if the crypto DB is wiped, the stored recovery key becomes stale. The setup wizard's "Wipe E2EE state" option clears this key. Re-running hermes gateway setup matrix or hermes gateway verify-matrix re-establishes trust.
---
How to test
Prerequisites:
- Synapse homeserver (or any CS API-compatible homeserver)
- Two Matrix accounts: one for the bot, one for yourself
- pip install "mautrix[e2be]" asyncpg aiosqlite base58 + sudo pacman -S libolm (or distro equivalent)
Basic flow:
# 1. Configure
hermes gateway setup matrix
# Enter homeserver URL, bot password (wizard handles the rest)
# 2. Start
hermes gateway restart
# 3. Invite the bot to a room in Element, send a message
# Bot should respond within a few seconds
E2EE verification:
# After setup wizard completes, restart Element
# Bot should appear as a verified device (shield icon)
# If not, run:
hermes gateway verify-matrix
Reconfigure (clean state):
hermes gateway setup matrix
# Select "Reconfigure Matrix?"
# Select "Wipe existing E2EE state?" — this clears crypto.db and all stored keys
# Re-run setup from scratch
---
Dependencies added
| Package | Why | Optional |
|---------|-----|----------|
| mautrix[e2be] | Matrix client + E2EE backend | Required for E2EE; mautrix alone for non-E2EE |
| asyncpg | SQL dialect for PgCryptoStore (used with aiosqlite) | Required for E2EE |
| aiosqlite | SQLite backend for PgCryptoStore (no PostgreSQL needed) | Required for E2EE |
| base58 | Recovery key encoding | Required for E2EE |
| libolm | C library for Olm/Megolm crypto (system package) | Required for E2EE |
None of these are installed by default — they are only needed when MATRIX_E2EE=true and are installed automatically by the setup wizard if requested.
---
Environment variables
| Variable | Required | Purpose |
|----------|----------|---------|
| MATRIX_HOMESERVER_URL | Yes | e.g. https://matrix.example.org |
| MATRIX_ACCESS_TOKEN | Yes | Bot account access token (syt_...) |
| MATRIX_USER_ID | Yes | Bot Matrix ID (@bot:example.org) |
| MATRIX_DEVICE_ID | Recommended | Pins to a fixed device; prevents session replay on restart |
| MATRIX_ALLOWED_USERS | Recommended | Comma-separated Matrix IDs allowed to message the bot |
| MATRIX_HOME_CHANNEL | Optional | Room ID for cron job delivery |
| MATRIX_VERIFY_SSL | Optional | false for self-signed TLS (default true) |
| MATRIX_E2EE | Optional | true to enable E2EE (requires dependencies above) |
| MATRIX_PASSWORD | Optional | Bot password for cross-signing bootstrap UIA |
| MATRIX_RECOVERY_KEY | Optional | Auto-saved by gateway; used for device self-signing on restart |
---
Test results
3533 passed, 35 failed (pre-existing), 157 skipped
All 35 failures are pre-existing on main and unrelated to this PR. The 76 new Matrix tests all pass.
