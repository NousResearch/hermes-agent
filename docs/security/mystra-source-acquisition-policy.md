# Mystra YouTube / transcript source-acquisition policy

Status: baseline policy + implemented offline foundation
Owner: Mystra (knowledge extraction) with Gond/Helm for Spearhead safety review
Approval provenance: Kanban gate `t_63cf2237` (Filip approval), implementation card `t_995513e8`.
Design provenance: `/home/filip/spearhead-execution/20260528-source-spikes/yt-dlp/followups/youtube-transcript-acquisition-hardening.md`
(derived from a source-only spike of yt-dlp at `acf8ab7a6e3024325f62426e35a17f365c4d5d54`; no yt-dlp code is imported or vendored).
Related: [Helm side-effect policy](helm-side-effect-policy.md).

## Purpose

Defines the default posture for Mystra acquiring sources and information from
public YouTube videos, channels, and playlists for knowledge extraction, and
the hard boundaries around authenticated/credentialed access. It pairs with the
implemented offline foundation in
`skills/media/youtube-content/scripts/acquisition.py`.

## What Filip approved (gate `t_63cf2237`)

- Public YouTube videos, channels, and playlists **may** be acquired,
  downloaded, and transcribed for Mystra knowledge extraction.
- Mystra **may** obtain sources/information from public/legitimate
  videos/channels/playlists autonomously.
- Authenticated YouTube access is allowed **only** when Filip explicitly
  approves that specific case.

## Hard boundaries (still gated — require a separate approval card)

- No credential capture/storage and no browser-cookie import.
- No paid APIs or spending.
- No proxy/VPN, CAPTCHA solver, or token/challenge-solver adapters.
- No Notion/Obsidian live writeback.
- Media downloads are allowed **only** as local acquisition/transcription
  inputs with provenance and retention notes — never for redistribution.

These map to the deny/approve tiers in the [Helm side-effect policy](helm-side-effect-policy.md):
cookie/credential/proxy/solver actions are S3–S5 and stay default-deny until an
authorized human approves the named case.

## Acquisition model (implemented)

The acquisition layer is a layered, default-anonymous state machine:

1. **Classify** the source (`classify_source`) into `video`, `channel`,
   `playlist`, `transcript`, `audio_media`, or `unsupported`. The raw URL is
   never retained — only a `sha256:` hash plus non-secret ids.
2. **Sequence providers** (`acquire`) in priority order. Each provider declares
   `supports()` and a typed `auth_mode`. Only `anonymous_public` runs by
   default; any other mode is skipped unless an `AuthPolicy` carries a matching
   approved `approval_id`.
3. **Classify outcomes** into the typed status taxonomy and **preserve partial
   success** (metadata can survive even when a transcript is blocked).
4. **Retry** only transient/throttle classes (`NETWORK_TRANSIENT`,
   `RATE_LIMITED`) with bounded exponential backoff + jitter, honoring
   `Retry-After`. Auth/CAPTCHA/geo/token/DRM/unsupported are never blindly
   retried; login/CAPTCHA/geo/stale-auth **hard-stop** the whole run with no
   automatic fallback.
5. **Emit redaction-safe evidence** (`AcquisitionResult.to_evidence`).

### Status taxonomy

`OK`, `PARTIAL_METADATA_ONLY`, `NO_TRANSCRIPT_AVAILABLE`, `LOGIN_REQUIRED`,
`AUTH_COOKIE_STALE`, `CAPTCHA_REQUIRED`, `RATE_LIMITED`, `GEO_RESTRICTED`,
`DRM_OR_MEDIA_BLOCKED`, `TOKEN_REQUIRED`, `NETWORK_TRANSIENT`, `UNSUPPORTED_URL`.

Each status has a default next-safe-action (`next_safe_action`) — e.g.
`LOGIN_REQUIRED` → stop and request explicit approval; `CAPTCHA_REQUIRED` →
stop, never auto-solve; `RATE_LIMITED` → back off, no credential workaround.

### Transcript provenance (no hallucination)

Results carry `TranscriptProvenance` distinguishing `manual`, `automatic` (ASR),
`translated`, and `live_chat`, plus original/requested/translated languages.
When the status is not `OK`, the layer forces provenance back to `none` so a
metadata-only or failed acquisition can never imply transcript text that was
not actually obtained.

## Redaction

`redact_acquisition_text` is the single chokepoint for any value bound for
Discord, Kanban, logs, artifacts, or prompts. It scrubs cookie headers and named
Google/YouTube session cookies (SAPISID, HSID, SSID, `__Secure-*`, `LOGIN_INFO`,
…), `poToken`, `visitorData`, Data Sync IDs, and authorization headers; hashes
signed `googlevideo.com` media URLs; and layers Hermes'
`agent.redact.redact_sensitive_text` for generic credential coverage. Evidence
objects pass every string artifact leaf through it, and the `redactions` field
records which categories were scrubbed without exposing values.

## Auth modes

`anonymous_public` (default, always allowed) · `approved_cookie_file` ·
`approved_oauth_or_api_key` · `browser_session_import` · `manual_human_browser`.
All non-anonymous modes are default-deny and require a current, named approval
(`AuthPolicy(mode=…, approved=True, approval_id=…)`). Stale approved auth returns
`AUTH_COOKIE_STALE` and stops — it never refreshes, rotates, or logs in.

## Activation status

The module and its tests are **implemented for review** and are not yet wired
into the live agent tool path. The existing `youtube-content` skill
(`scripts/fetch_transcript.py`) continues to handle public transcript fetches;
this hardening layer is the foundation a future, separately-approved card can
adopt as the acquisition entry point. No live YouTube scraping, media download,
browser login, or credential access is performed by this code.

## Tests

`tests/skills/test_acquisition.py` — 49 offline tests covering URL
classification, public transcript success, ASR/manual/translated provenance,
playlist/channel expansion metadata, disabled/no transcript, login-required and
CAPTCHA/geo hard stops, auth default-deny (cookie/browser providers not selected
without matching approval), retry/backoff (transient succeeds/exhausts,
non-retryable stops immediately, `Retry-After` honored), redaction, and
no-hallucination partial-source handling. No network, media, or credentials.
