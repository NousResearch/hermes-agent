feat(gateway): add Threema platform adapter (end-to-end encrypted mode)

## What does this PR do?

Adds a **Threema** gateway adapter — the privacy-first Swiss messenger requested in #21307 — as a bundled platform plugin under `plugins/platforms/threema/`. Outbound goes through `/send_e2e`; inbound arrives as MAC-verified callbacks that are opened with NaCl. Text, files, images, audio, video and location messages work in both directions, cron delivery works even when the job runs detached from the gateway, and nothing in the agent core changes.

**Why a plugin, not the three core files the issue sketches.** The issue's file list (`gateway/platforms/threema.py` + a `Platform` enum member + an `elif` in `gateway/run.py`) describes the built-in path, and `gateway/platforms/ADDING_A_PLATFORM.md` then lists **sixteen** more integration points it implies — auth maps, the cron `platform_map`, `send_message_tool` routing, `PLATFORM_HINTS`, the toolset composite, `hermes_cli/status.py`, the setup wizard. That path is no longer how platforms are added: every platform, Telegram and Discord included, now lives in `plugins/platforms/`, and `Platform("threema")` is created on demand by the enum's `_missing_` from the bundled-plugin directory scan. Registering through `ctx.register_platform()` with `cron_deliver_env_var`, `standalone_sender_fn`, `allowed_users_env`, `allow_all_env`, `env_enablement_fn`, `setup_fn` and `platform_hint` covers all sixteen points at once. The one core line in the diff is a row in the `tools/lazy_deps.py` table, which is where every platform plugin declares its installable dependencies (`platform.wecom_callback` is the precedent).

**Why PyNaCl instead of the official SDK.** The issue recommends `threema.gateway` and flags its libsodium requirement as the dependency wart. Checking what 8.0.0 actually pulls: `logbook<2`, `libnacl<3`, `click<9`, `aiohttp`, `wrapt`. `libnacl` is a ctypes binding over a *system* libsodium — that is the wart, and it is not fixable from our side — while `logbook` drops a second logging framework into a process that already has one, and `click` a second CLI framework. PyNaCl 1.6.2 publishes 24 binary wheels (manylinux, macOS universal2, Windows) with libsodium statically linked, needs only `cffi`, and is **already pinned in this repo** (`pynacl>=1.6,<1.7` in `[tool.uv]` overrides, because `discord.py[voice]` pulls it). So the E2E container is implemented directly — about 200 lines — against Threema's published spec, and the install story becomes "pip install pynacl", with no `apt install libsodium-dev` step in the setup guide.

**Spec details that a summary loses, and that the implementation pins with tests:**

- **Padding is not plain PKCS#7.** The random 1..255 pad is *widened* so the padded plaintext is at least 32 bytes, which is what stops a short message from leaking its length. A naive implementation is wire-compatible right up until someone sends "ok".
- **The callback MAC** is `HMAC-SHA256(from || to || messageId || date || nonce || box, secret)` over the fields exactly as POSTed — hex still hex — compared with `hmac.compare_digest` **before** anything is parsed or decrypted.
- **Retry semantics drive three decisions.** Threema retries a non-200 callback three times at five-minute intervals and then discards the message. So the dedup window (30 min) outlives the whole retry schedule; an authentic-but-undecryptable box is **acked**, because retrying a box this key pair can never open just repeats the failure four times; and only a transient API failure answers 500 on purpose.
- **The text cap is 3500 bytes, not characters.** 1200 emoji are 4800 bytes and would be rejected by a character-counting chunker. Splitting counts UTF-8 bytes and never cuts a code point.
- **Blob nonces are protocol constants** (23 zero bytes then `0x01` for the file, `0x02` for the thumbnail), verified against the official SDK's source. Blobs are encrypted before upload and the file container carries the symmetric key.
- **Delivery receipts (`0x80`) are dropped, not dispatched.** They are the recipient's checkmarks; forwarding them would wake the agent on every read confirmation.
- **The single and bulk endpoints disagree**: `/send_e2e` takes hex, `/send_e2e_bulk` takes base64. This uses the single endpoint and hex.

**Threema costs money, and the adapter treats that as a design constraint.** Each message is a credit and a blob upload is a second one. So the typing indicator is a no-op rather than a fake keepalive, delivery receipts are not sent back, splitting a long answer logs how many credits it just spent, and the platform hint tells the model that every message it sends costs the user money and that Threema renders no Markdown.

**The TLS wart is surfaced, not papered over.** Threema delivers only to an HTTPS URL with a publicly trusted certificate and offers no API to set that URL. `connect()` warns when `public_url` is missing or not HTTPS, the local listener exposes `/health` so the proxy side can be checked independently, and the setup guide and the wizard's closing message both spell out the two manual panel steps.

**Security.** The API secret travels in query parameters, so every logged or raised URL is passed through a redactor first; the private key is read from a file rather than the process environment; the callback body is capped at 64 KiB before any crypto runs (`client_max_size` plus the cap); remote image URLs are fetched through `tools/url_safety`'s SSRF-safe client; the platform is registered `pii_safe=True`.

## Related Issue

Fixes #21307

## Type of Change

- [ ] 🐛 Bug fix (non-breaking change that fixes an issue)
- [x] ✨ New feature (non-breaking change that adds functionality)
- [ ] 🔒 Security fix
- [ ] 📝 Documentation update
- [ ] ✅ Tests (adding or improving test coverage)
- [ ] ♻️ Refactor (no behavior change)
- [ ] 🎯 New skill (bundled or hub)

## Changes Made

- **`plugins/platforms/threema/crypto.py`** (new, 234 lines) — key parsing (`private:<hex>` panel format or bare hex, read from a file), PKCS#7 padding with the 32-byte floor, container encrypt/decrypt, blob secretbox with the fixed protocol nonces, and the callback MAC with a constant-time verifier. Pure functions over `bytes`; PyNaCl is the only third-party import.
- **`plugins/platforms/threema/api.py`** (new, 210 lines) — the REST client: `/send_e2e`, `/upload_blob`, `/blobs/{id}`, `/pubkeys/{id}` (cached 24 h, as Threema asks), `/capabilities/{id}`, `/credits`, `/lookup/phone`, `/lookup/email`. Status codes carry the cause — 402 is "out of credits, every send costs one", 413 is the 7812-byte box cap, 429 is rate-limiting — and the box-size check happens client-side so an oversized message never spends a credit to be rejected. The transport is injectable, which is how the tests exercise real request shapes without a network.
- **`plugins/platforms/threema/adapter.py`** (new, 696 lines) — `ThreemaAdapter`: the aiohttp callback receiver (registered through `shared_ingress.bind_listener`, so a multiplex secondary is served at `/p/<profile>/…` instead of binding its own port), byte-accurate chunked sending, blob-backed image/file/audio/video sending, inbound file download-and-decrypt into the media cache, location rendering, dedup, and the `register(ctx)` hooks. The callback *decision* lives in `process_callback(form) -> (status, event)` with no aiohttp in it, so the security-critical branches are testable on a bare install.
- **`plugins/platforms/threema/plugin.yaml`** (new) — manifest with `requires_env` / `optional_env` rich entries, which auto-populate the setup wizard's prompts and password flags.
- **`tools/lazy_deps.py`** — one row, `"platform.threema": ("pynacl==1.6.2", "aiohttp==3.14.3")`, with a comment recording why it is PyNaCl and not the SDK.
- **`tests/gateway/test_threema.py`** (new, 42 tests, no network) — see below.
- **Docs** — `website/docs/user-guide/messaging/threema.md` (setup guide, including the TLS requirement, the nginx snippet, the cost model and a troubleshooting table), the capability row and Next Steps link in `messaging/index.md`, a `### Threema` block in `reference/environment-variables.md`, and the sidebar entry.

No changes to the agent core, the toolset schema, `gateway/config.py`, `gateway/run.py`, `cron/scheduler.py`, or `tools/send_message_tool.py`.

## How to Test

**Automated:**

```bash
scripts/run_tests.sh tests/gateway/test_threema.py -q        # 42 passed
scripts/run_tests.sh tests/gateway -q                        # 7427 passed, 119 failed
scripts/run_tests.sh tests/cron tests/hermes_cli/test_plugin_catalog.py -q   # 1277 passed
ruff check plugins/platforms/threema tests/gateway/test_threema.py           # All checks passed!
```

The 119 failures in `tests/gateway` are pre-existing in this environment (optional SDKs — slack, discord, aiohttp — are not installed). I captured the failing-test list on `main` and on this branch and diffed them: **identical**, and the passing count is baseline + 42. That diff is also how I caught the one real contract this first missed — `test_all_connectable_adapters_wire_plugin_handlers` requires every adapter to call `_wire_plugin_handlers` in `connect()` so plugins can register routes before the router freezes; it is wired now, and the run went from 120 failures back to the baseline 119.

What the tests pin, chosen to be contracts rather than snapshots:

| Area | Examples |
|---|---|
| Container | padding widened to 32 bytes; round trip carries the type byte; a wrong key fails; blob nonces are the protocol constants and a thumbnail blob will not open as a file blob |
| Callback auth | MAC matches the documented field order byte for byte; a swapped body, a wrong secret, a missing MAC and an unconfigured secret are all rejected |
| Retry semantics | a replayed callback is delivered once but still acked 200; an undecryptable box is acked; a transient API failure returns 500; the dedup TTL is asserted to exceed the 3 × 5 min retry schedule |
| Billing-sensitive behaviour | an oversized box is refused before the request; a long answer reports every part; typing sends nothing |
| Wire shapes | `/send_e2e` carries hex nonce and box; `/upload_blob` puts auth in the query and the blob in multipart; the uploaded bytes are *not* the plaintext and decrypt with the key from the container |
| Plumbing | `Platform("threema")` resolves with no core edit; `config.yaml` → `PlatformConfig.extra` → connected platform → adapter fields, over the real loader and registry |
| Registration | `register()` declares every hook; the hint warns about Markdown and about credits |

**Against a real Gateway ID** (end-to-end mode; basic mode cannot receive):

```bash
cat > ~/.hermes/threema_private.key <<'KEY'
private:<64 hex characters from the Gateway panel>
KEY
chmod 600 ~/.hermes/threema_private.key

cat >> ~/.hermes/.env <<'ENV'
THREEMA_GATEWAY_ID=*MYBOT
THREEMA_API_SECRET=...
THREEMA_PRIVATE_KEY_PATH=~/.hermes/threema_private.key
THREEMA_PUBLIC_URL=https://hermes.example.com
THREEMA_ALLOWED_USERS=ECHOECHO
ENV

hermes gateway start
curl -s http://127.0.0.1:8647/health
# {"status": "ok", "platform": "threema", "identity": "*MYBOT"}
```

Point the Gateway panel's callback URL at `https://<host>/threema/callback` (a reverse proxy with Let's Encrypt, a Cloudflare Tunnel, or `ngrok` for dev), then message the ID from the Threema app. Worth checking specifically:

1. A reply longer than 3500 bytes arrives as several messages and the log names the credit cost.
2. Sending a photo from the app arrives as a media attachment the agent can see (blob downloaded and decrypted).
3. Reading the reply in the app does **not** produce a second agent turn (the delivery receipt is dropped).
4. Deliberately breaking `THREEMA_API_SECRET` produces `Callback MAC verification failed` and no decryption attempt.
5. `hermes cron` with `deliver=threema` delivers even when the job runs detached from the gateway.

## Checklist

### Code

- [x] I've read the [Contributing Guide](https://github.com/NousResearch/hermes-agent/blob/main/CONTRIBUTING.md) and `gateway/platforms/ADDING_A_PLATFORM.md`
- [x] My commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat(gateway): …`)
- [x] I searched for existing PRs to make sure this isn't a duplicate — no open PR and no plugin mentions Threema
- [x] My PR contains **only** changes related to this feature (one commit)
- [x] I've run the tests — via `scripts/run_tests.sh` rather than bare `pytest`, as `AGENTS.md` requires for CI parity: `tests/gateway` failure list is byte-identical to `main`'s, plus 42 new passing tests
- [x] I've added tests for my changes — behaviour contracts, no change-detectors; the one source-reading assertion in the suite is the repo's own pre-existing wiring invariant, which this now satisfies
- [x] I've tested on my platform: Ubuntu 26.04.1 LTS, kernel 7.0.0-31-generic, Python 3.11.16

### Documentation & Housekeeping

- [x] I've updated relevant documentation — new messaging guide, index capability row and Next Steps link, environment-variable reference, sidebar
- [x] `cli-config.yaml.example` — N/A: the platform's settings live under `gateway.platforms.threema.extra`, which the generic loader already accepts; no new top-level config keys
- [x] `CONTRIBUTING.md` / `AGENTS.md` — N/A: no architecture or workflow change; the plugin follows `ADDING_A_PLATFORM.md`'s plugin path as written
- [x] I've considered cross-platform impact — pure Python plus PyNaCl, which ships wheels for Linux, macOS (universal2) and Windows; no POSIX-only primitives, no shell-outs, paths via `pathlib`
- [x] Tool descriptions/schemas — N/A: no tool changed. The platform hint is per-session prompt text, not a tool schema, so the core toolset is untouched

### For a New Platform

- [x] Registered through `ctx.register_platform()` with no core adapter factory, enum member, or authorization-map edit
- [x] `check_fn` is a passive probe and the active installer is `ensure_deps_fn` — the split that keeps status displays from pip-installing and lets `create_adapter()` install right before connect. Every module-level availability flag the install affects is rebound, so a `check_fn` that was False at import does not stay False after a successful install
- [x] Inbound port binding goes through `shared_ingress.bind_listener`, so a multiplex secondary is served at `/p/<profile>/…` instead of fighting for the port
- [x] Self-messages cannot loop (the Gateway ID only receives; its own sends never come back as callbacks) and delivery receipts are filtered
- [x] Sensitive identifiers are redacted in logs; the API secret never reaches a log line or an exception message

## Screenshots / Logs

Startup, with the TLS warning that fires when the callback URL is not configured:

```
[Threema] Connected as *MYBOT (412 credits), listening on *:8647/threema/callback
[Threema] No public_url configured. Threema only delivers to an HTTPS URL with a publicly
          trusted certificate — set the callback URL in the Gateway panel to
          https://<your-host>/threema/callback (reverse proxy, Cloudflare Tunnel, or ngrok for dev).
```

A forged callback, rejected before any decryption:

```
[Threema] Callback MAC verification failed (from=ECHOECHO)
```

A long answer, with the cost stated:

```
[Threema] Message split into 3 parts (3 credits)
```

Test run:

```
=== Summary: 1 files, 42 tests passed, 0 failed (100% complete) in 6.4s (64 workers) ===
=== Summary: 832 files, 7427 tests passed, 119 failed, 50 skipped (100% complete) in 98.8s (64 workers) ===
    (119 = the pre-existing baseline on main; the failing-test lists are identical)
```
