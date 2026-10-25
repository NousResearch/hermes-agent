# WhatsApp Feature Parity — Coverage Matrix & Gap Catalog (living doc)

Campaign: **WhatsApp Feature Parity & Alignment** — upstream meta-issue [#79890](https://github.com/NousResearch/hermes-agent/issues/79890).

Source analyzed: **NousResearch/hermes-agent** `main` @ commit `aec3318` (verified 2026-08-06).
This document is the **living parity matrix**: it records, per capability, the current
coverage status, the evidence anchor (`file:line`), the numbered gap ID, its severity, and
its tracking issue interlock. It is the Phase‑0 deliverable that later phases read.

> This is a **discovery/catalog** document. It makes no code changes. Code changes land in
> later phases via fork PRs (head `Shotflame:branch` → upstream `NousResearch:main`).

## Legend

- **Supported** = implemented, code + docs evidence, works as intended.
- **Partial** = implemented but with a known gap / one backend only / degraded.
- **Missing** = not implemented (no code path; sibling platforms have it).
- **Unknown** = no evidence found; unverified.

Backends: **B** = Baileys bridge (`hermes whatsapp`, `plugins/platforms/whatsapp/adapter.py`
+ `scripts/whatsapp-bridge/bridge.js`), **C** = Meta Business Cloud API
(`hermes whatsapp-cloud`, `gateway/platforms/whatsapp_cloud.py`).

## Gap ID classification

| GAP ID prefix | Meaning |
|---|---|
| `GAP_UNSUPPORTED` | Capability Missing — no code path. |
| `GAP_PARTIAL` | Capability Partial — implemented but degraded / one backend only. |
| `GAP_CONFLICTED` | Doc and code disagree (doc claims absent, code has it). |
| `GAP_DOCS` | Docs-only gap (needs documentation correction). |
| `GAP_BUG_TRACKED` | Capability Missing with an existing upstream tracker issue. |

Every gap below has **one** tracking card (interlock `I=card`) and a **file:line evidence
anchor**. Zero duplicates: each gap appears exactly once.

---

## ⚠ P1 — Security / data-loss findings (surface first)

Triage standard (campaign DoD): security and data-loss findings are **P1** and surface
before feature work.

| GAP ID | Finding | Evidence anchor | Severity | Interlock |
|---|---|---|---|---|
| `GAP_UNSUPPORTED_SESS` | **Pairing / session credential storage.** WhatsApp session `creds.json` holds the authenticated Baileys session (auth keys/device creds) at `~/.hermes/platforms/whatsapp/session/creds.json`. A loss/exfiltration of this file = full WhatsApp account compromise for the paired device. No explicit perms hardening or secret-wrapping documented. | `plugins/platforms/whatsapp/adapter.py:541` (`creds_path = self._session_path / "creds.json"`); `adapter.py:426-428` (session path default) | **P1** | card `t_60190daf` (ledger reconcile) |
| `GAP_DOCS_PHONEPII` | **Phone-number PII identity handling.** `gateway/whatsapp_identity.py` canonicalises phone-form JIDs (`15551234567@s.whatsapp.net` → numeric); phone numbers are user-config/allowlist identity (`pairing.py:296` mirrors a normalized phone into `WHATSAPP_ALLOWED_USERS`). Phone PII is processed/logged; any data-loss path (logs, session, backups) leaks phone numbers. | `gateway/whatsapp_identity.py:7,52-54`; `gateway/pairing.py:296`; `gateway/platforms/whatsapp_cloud.py:222,365` (`phone_number_id`) | **P1** | card `t_60190daf` (ledger reconcile) |
| `GAP_DOCS_TOKEN` | **Cloud access-token / app-secret credential surface.** Meta Cloud `WHATSAPP_CLOUD_ACCESS_TOKEN` (permanent System User token), `WHATSAPP_CLOUD_VERIFY_TOKEN`, and `app_secret` are required config; inbound refused with 503 without secret. Token/app-secret in env must be treated as P1 secrets (no log exposure, no commit). | `gateway/platforms/whatsapp_cloud.py:222-229,482-492,385`; `whatsapp-cloud.md` §Prerequisites | **P1** | card `t_60190daf` (ledger reconcile) |

> P1 items are surfaced first per the issuer's hard standard. They are catalogued in Phase 0
> (`this document`) and owned by the ledger-reconcile card `t_60190daf` → tracked through the
> Phase 2/3 pipeline.

---

## Lane S1 — Core messaging

| Capability | Status | Evidence anchor | GAP ID | Severity | Interlock |
|---|---|---|---|---|---|
| Text send / markdown / code blocks | Supported B/C | `bridge.js:823`; `whatsapp_common.py:74-75` | — | — | — |
| Streaming responses | Supported | `whatsapp.md`; TIER_MEDIUM gateway | — | — | — |
| Chunking @ 4096 | Supported B/C | `whatsapp_common.py:74` | — | — | — |
| Text debounce batching | Supported B | `whatsapp.md` §Message Batching | — | — | — |
| Images / documents / video / audio-in / voice-in (STT) | Supported B/C | `bridge.js:917,947,981`; `whatsapp_cloud.py:_send_media` | — | — | — |
| **Voice-out (TTS voice note)** | **Partial** | C: `whatsapp_cloud.py:15,132,140` (mmpeg+MP3 fallback); B: `adapter.py:1242 send_voice` (MP3 attachment) | `GAP_PARTIAL_VOICEOUT` | P2 | card Ph1 `t_650cd5be` (conformance) |
| **Sticker (outbound)** | **Partial** | inbound caps `whatsapp_cloud.py:117`; outbound path unverified | `GAP_PARTIAL_STICKER` | P3 | card Ph1 `t_a5435728` |
| **Reactions (send/mirror)** | **Missing** | no `send_reaction` in WhatsApp adapter/`bridge.js`; siblings have it (`matrix/adapter.py:3731`, `buzz/adapter.py:641`, `signal.py:1558`) | `GAP_BUG_TRACKED_REACT` | P2 | card `t_34577059` (#23899) |
| **Edit messages** | **Partial (B only)** | B: `adapter.py:985 edit_message` → `bridge.js:862` (`/edit`); C: none | `GAP_PARTIAL_EDIT` | P2 | card Ph1 `t_3fb14ad6` |
| **Delete messages** | **Missing** | no delete path in WhatsApp adapter/bridge | `GAP_UNSUPPORTED_DELETE` | P2 | card `t_49f91f6b` |
| Typing indicator | Supported B/C (**doc conflict**) | B: `adapter.py:1268`, `bridge.js:1041` (`/typing`); C: `whatsapp_cloud.py:607` | `GAP_CONFLICTED_TYPING` / `GAP_DOCS_TYPING` | P2 | card Ph1 `t_5eb15fc2` |
| Read receipts | Supported B/C (**doc conflict**) | B: `adapter.py:453-458,1357 _send_read_receipt`; C: `whatsapp_cloud.py:594-630` | `GAP_CONFLICTED_READRC` / `GAP_DOCS_READRC` | P2 | card Ph1 `t_5eb15fc2` |
| **Location pins** | **Partial (B only)** | B: `adapter.py:1153 send_location`, `bridge.js:1019` (`/send-location`) + inbound; C: none | `GAP_PARTIAL_LOCATION` | P2 | card Ph1 `t_807c73ca` |
| **Contact / vCard share** | **Missing** | no contact handler found | `GAP_UNSUPPORTED_VCARD` | P3 | card `t_782c86f6` |
| Reply context (quoted replies) | Supported B/C | `whatsapp_common.py`; C inbound `context` (`whatsapp_cloud.py:582`) | — | — | — |
| **Group chats** | **Partial** | B: full group; C: **DMs only v1** (docs) | `GAP_PARTIAL_GROUPC` | P2 | card `t_28a48424` |
| **Ephemeral outbound** | **Partial** | inbound unwrap only (`bridge.js:230 ephemeralMessage`); no outbound control | `GAP_PARTIAL_EPHEMERAL` | P3 | card `t_2430213f` |

## Lane S2 — Business Cloud API surface

| Capability | Status | Evidence anchor | GAP ID | Severity | Interlock |
|---|---|---|---|---|---|
| Webhooks (verify + messages) | Supported C | `whatsapp_cloud.py:467-468,492,385` | — | — | — |
| App-secret signature check | Supported C | `whatsapp_cloud.py:492,385` (503 w/o secret) | — | — | — |
| **Message templates (24h escape)** | **Missing** | docs "Message-template support … not yet implemented"; Graph re-engagement | `GAP_BUG_TRACKED_TEMPLATES` | P1 | card `t_eacc7550` (#45935) |
| 24-hour conversation window | Partial | Meta rule acknowledged; auto-resolve depends on (missing) templates | `GAP_PARTIAL_24H` | P2 | card `t_eacc7550` |
| Outbound media size limits | Supported C | `whatsapp_cloud.py:113-117` | — | — | — |
| **Client-side outbound rate limiting** | **Missing** | docs "Hermes doesn't currently enforce this client-side" (Meta default 80/s) | `GAP_UNSUPPORTED_RATELIMIT` | P2 | card `t_f158014d` |
| **Phone-number provisioning** | **Partial** | Meta-side only; wizard prints B.Mgr links; no Hermes automation | `GAP_PARTIAL_PROVISION` | P3 | card `t_48bd3707` |
| **Quality rating** | **Unknown** | Meta dashboard only; surfaced in webhook parse (`whatsapp_cloud.py:1596`) not actionable | `GAP_UNSUPPORTED_QUALITY` | P3 | card `t_1ddb604d` |

## Lane S3 — Groups

| Capability | Status | Evidence anchor | GAP ID | Severity | Interlock |
|---|---|---|---|---|---|
| **Group metadata** | **Partial** | bridge `sock.groupMetadata()` (`bridge.js:1088`) internal only | `GAP_PARTIAL_GROUPMETA` | P3 | card `t_28a48424` |
| **Mention gating (@)** | **Partial** | `adapter.py:458 _compile_mention_patterns`; config `require_mention`; #7269 "incomplete" | `GAP_BUG_TRACKED_MENTION` | P3 | card `t_b140e1bf` (#7269) |
| **Per-sender group gating** | **Partial** | B `group_policy` incl `pairing` + `group_allow_from`; #48394 class | `GAP_PARTIAL_PERSENDER` | P3 | cards `t_eb95a23d` (#48394), `t_7b7d5b78` (#38710/#41989) |
| **Group admin ops / invite links** | **Missing** | no admin/invite feature in adapter/bridge | `GAP_UNSUPPORTED_ADMIN` | P2 | card `t_30f1c860` |
| Broadcast / Channel / Newsletter filter | Supported B/C | `whatsapp_common.py`; `bridge.js:561-564` | — | — | — |
| **Silent / read-only monitor** | **Partial** | mention-gate exists; #38710/#41989 DM-only control gaps | `GAP_BUG_TRACKED_SILENT` | P3 | card `t_5a02e8c1` (#33912) |

## Lane S4 — Bridge / backend lifecycle

| Capability | Status | Evidence anchor | GAP ID | Severity | Interlock |
|---|---|---|---|---|---|
| QR pairing | Supported B | `hermes whatsapp`; docs §Step 1 | — | — | — |
| **QR dark-mode inversion** | **Partial** | wizard QR only | `GAP_BUG_TRACKED_QRDM` | P4 | card `t_d55557e5` (#58038) |
| Session persistence & re-pair | Supported B | session `~/.hermes/platforms/whatsapp/session`; `test_whatsapp_bridge_pidfile.py` | — | — | — |
| **Reconnect / backoff health** | **Partial** | reconnect logic + tests; health-flap #63277 | `GAP_BUG_TRACKED_RECONNECT` | P2 | card `t_1100838b` (#63277) |
| **Multi-device semantics** | **Partial** | Baileys multi-device; contested #7274 | `GAP_BUG_TRACKED_MULTIDEV` | P3 | card `t_d52149cb` (#7274) |
| **Node subprocess hygiene** | **Partial** | process-group spawn `adapter.py:670`; Windows detach #68128, console-flicker #75628 | `GAP_BUG_TRACKED_NODEHYG` | P2 | card `t_1882e118` (#68128/#75628) |
| **Neonize pure-Python (no Node)** | **Missing** | #7274 (contested); docs require Node 20+ | `GAP_BUG_TRACKED_NEONIZE` | P3 | card `t_d52149cb` (#7274) |

## Lane S5 — Rich features

| Capability | Status | Evidence anchor | GAP ID | Severity | Interlock |
|---|---|---|---|---|---|
| Interactive buttons/lists | Supported C / Partial B | C: `interactive.type` (`whatsapp_cloud.py:757,800`); B text-fallback | `GAP_PARTIAL_BUTTONS` | P3 | card `t_650cd5be` (conformance) |
| **Native polls** | **Partial (B only)** | B: `send-poll` `bridge.js:997`, `adapter.py:1067`; C: none; #38892 | `GAP_BUG_TRACKED_POLLS` | P3 | card `t_23d71ed0` (#38892) |
| Approval / slash-confirm buttons | Supported C | `whatsapp_cloud.py:845,903` | — | — | — |
| **Flows / Payments / Catalogs** | **Missing** | no evidence | `GAP_UNSUPPORTED_FPC` (M5 scope gate) | P2 | card `t_246ca15b` (M5) |
| **History backfill** | **Missing** | — | `GAP_BUG_TRACKED_BACKFILL` | P3 | card `t_06a1725d` (#42718) |
| **Emoji reactions-as-replies** | **Missing** | — | `GAP_BUG_TRACKED_EMOJI` | P3 | card `t_cc0ed163` (#60736) |
| **Status / broadcast controllability** | **Partial** | filtered from processing (`bridge.js:564`), not controllable | `GAP_PARTIAL_STATUS` | P3 | card `t_8f2c77b5` |

## Lane S6 — Decomposition / headroom

| Capability | Status | Evidence anchor | GAP ID | Severity | Interlock |
|---|---|---|---|---|---|
| **adapter.py size control** | **Gap (addressable)** | `adapter.py` = **1,918 ln** (96% of 2,000-ln ceiling); no main worktree/PR yet | `GAP_UNSUPPORTED_S6` | P2 | card `t_6548dfa5` |

---

## Doc/code conflict (flagged for correction)

`website/docs/user-guide/messaging/whatsapp-cloud.md` comparison table claims **Baileys has
no typing indicator and no read receipts**, but the plugin adapter + Node bridge implement both:

- Typing: `adapter.py:1268` + `bridge.js:1041` (`/typing`) — B; `whatsapp_cloud.py:607` — C.
- Read receipts: `adapter.py:453-458,1357` (`_send_read_receipt`) — B; `whatsapp_cloud.py:594-630`
  (blue checkmarks) — C.

GAP IDs: `GAP_CONFLICTED_TYPING`, `GAP_CONFLICTED_READRC` (code/doc conflict) and
`GAP_DOCS_TYPING`, `GAP_DOCS_READRC` (docs correction). Interlock: Ph1 docs card `t_5eb15fc2`.

---

## References & sources

- Docs: `website/docs/user-guide/messaging/whatsapp.md`, `.../whatsapp-cloud.md`, `.../index.md`
- Code: `plugins/platforms/whatsapp/adapter.py`, `gateway/platforms/whatsapp_cloud.py`,
  `gateway/platforms/whatsapp_common.py`, `gateway/whatsapp_identity.py`,
  `scripts/whatsapp-bridge/bridge.js`
- Config/platform: `hermes_cli/platforms.py`, `hermes_cli/subcommands/whatsapp.py`,
  `hermes_cli/setup_whatsapp_cloud.py`, `gateway/config.py`, `toolsets.py`, `.env.example`
- Tests: 21 `test_whatsapp*` files; `tests/conformance/vectors/whatsapp.json` (44 vectors)
- Tracker refs: #23899, #7269, #45935, #38892, #48394, #38710, #41989, #63277, #68128,
  #75628, #7274, #42718, #60736, #7992, #33912, #58038, #79890 (meta)

*Maintained by the WhatsApp Feature Parity campaign (board: `t_0e530d03` catalog,
`t_60190daf` ledger, `t_81e69d77` tracking). Update this file when coverage or evidence
changes.*
