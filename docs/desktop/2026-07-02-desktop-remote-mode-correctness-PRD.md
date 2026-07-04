# PRD: Hermes Desktop Remote-Mode Correctness — /compress, HTML/file open, and the remote-path audit

- **Status:** v1.2 — super-passed (pass 8) + Phase-0 live repro complete + pass-9 D-9 delta folded. **BUILD UNBLOCKED.**
- **Author:** Apollo, 2026-07-02
- **Repo:** `Kyzcreig/hermes-agent` (fork), `apps/desktop` + `tui_gateway`
- **Prior art:** PR #167 (`fix(desktop): read composer image preview locally in remote mode`, merged `1184b0259`) — same bug class, fixed for composer-image preview. This PRD covers the remaining instances.

## 1. Summary & Goal

Ace uses Hermes Desktop **primarily in remote mode**: thin client on the MacBook Pro
(`connection.json: {"mode":"remote","remote":{"url":"http://mac-studio-m3u:9119","authMode":"oauth"}}`)
driving the gateway on the Mac Studio. A whole class of desktop features silently assumes
client == backend filesystem/process. Two user-hit bugs this week (image-paste preview —
fixed; HTML attachment open + `/compress` — this PRD), plus an audit that found more.

**Goal:** every user-facing desktop surface behaves correctly in remote mode: files that
live on the gateway are fetched/rendered through the authenticated API, and session-level
commands operate on the LIVE gateway session — or fail loudly with an honest message.
Never a silent wrong-machine operation.

## 2. Non-Goals

- No new fork/repo. The desktop app lives in `apps/desktop` of the existing
  `Kyzcreig/hermes-agent` fork — "our own fork of hermes desktop" already exists by
  construction. (Called out because the user asked; resolved in D-1.)
- Not fixing auto-compaction tuning/thresholds (owned by `compression.*` config; separate).
- Not building generic client-side caching of gateway files.
- Not touching the protected narrow waist (model tool schema, prompt cache, agent loop).
- Upstreaming: each fix that applies cleanly on `origin/main` gets a follow-up upstream PR
  (like #56827), but upstream acceptance is not a gate for our fleet fix.

## 3. Ground truth (measured, 2026-07-02)

All line numbers at `fork/main` @ `1184b0259`.

### BUG-A — `/compress` in remote desktop is a three-part failure (user-hit)

Observed output (screenshot): the *warning* line shows the **live** session
("No changes from compression: 355 messages / ~430,420 tokens (unchanged)"), the *output*
line shows the **slash-worker subprocess** ("(._.) Not enough conversation to compress
(need at least 4 messages)" — `cli.py:9235`), and a final
`error: not a quick/plugin/skill command: compress` (`tui_gateway/server.py:11733`).

Mechanics, from source:
1. Desktop `slash.ts` sends `/compress` to `slash.exec` (`tui_gateway/server.py:12512`).
2. `slash.exec` runs the command in a **`_SlashWorker` subprocess** (`server.py:265`) — a
   separate `HermesCLI(resume=session_key)` whose `conversation_history` is hydrated from
   disk at worker spawn and is NOT the live session. It sees <4 messages → the "(._.)"
   line. **The worker's compress operates on the wrong history.**
3. `_mirror_slash_side_effects` (def `server.py:12418`; compress branch `:12450`; the
   `_compress_session_history` call `:12472`) then ALSO compresses the **live**
   session and returns its feedback as `warning` — so the user sees two contradictory
   reports from two different histories in one response.
4. The live mirror compress reported **"No changes"** at 355 msgs / 430K tokens — i.e. the
   live compression itself did nothing. **Phase-0 hypothesis (from live logs, NOT yet
   confirmed as THE cause):** `dashboard.err.log` shows
   `⚠ Skipping concurrent compression — another path is already compressing this session`
   and the concurrent-compression lock in `agent/conversation_compression.py`
   (sid-key `:485`, acquire `:512`, skip-and-return-unchanged `:530-556` — the canonical
   citation; the Phase-0 addendum refers to this same lock)
   **returns `messages` unchanged** when it can't acquire the lock ("returning messages
   unchanged to avoid session fork"). BUG-A's own double-execution (worker + mirror both
   compressing the same session) is a plausible *source* of that contention → a self-inflicted
   "No changes". A second candidate: the TUI-gateway compress path calls `_compress_context`
   **without `force=True`** (the `agent._compress_context(...)` call at `server.py:2855`,
   inside `_compress_session_history` — see the Phase-1 scope table), unlike CLI
   (`cli.py:9316`) and the manual path,
   so it does NOT bypass the summary-failure cooldown. Phase 0 must isolate which (or a third
   cause) actually fired, and record the *resolved* `compression.enabled` value, not just the
   code path. **This is BUG-A(b) and is scoped separately from the routing bug — see D-6.**
5. The `4018` error is the desktop's `command.dispatch` fallback (`slash.ts:205-214`):
   when `slash.exec` throws, the desktop retries via `command.dispatch`, which has **no
   `compress` case** (its ladder: qcmds/plugin/skill/learn/moa/retry/steer/goal/undo/snapshot
   → `_err 4018`). Exact trigger sequence for the throw needs the Phase-0 repro (suspects:
   second invocation while worker busy; worker death after first run).

Net effect: **the user's context was not compressed**, and the UI reported three
mutually-contradictory things.

### BUG-B — HTML/file artifact "open" routes to the CLIENT filesystem (user-hit)

Observed: clicking an HTML attachment → toast "Preview unavailable / No application found
to open URL" (the subtitle is macOS's own error string, not an i18n key).

Mechanics, from source:
1. `openExternalUrl` (`apps/desktop/electron/main.cjs:936`) handles `file://` by
   `shell.openPath(localPath)` **on the client**. Gateway-side files don't exist there.
2. `openPreviewInBrowser` (`main.cjs:1007`) same: `file:` branch → local
   `shell.openExternal(pathToFileURL(localPath))`.
3. `preview-row.tsx openInBrowser` builds the URL via `normalizeOrLocalPreviewTarget` →
   `pathToFileUrl(path)` (`local-preview.ts`) — a client-local `file://` URL for a
   gateway path — then hands it to `openPreviewInBrowser`. Broken by construction remotely.
4. The right-rail HTML preview pane renders `<webview src={target.url}>`
   (`preview-pane.tsx:519`) with that same client-local `file://` URL → blank/fail remotely.
5. **Partial remote-awareness exists and must be reused, not duplicated:**
   `mediaExternalUrl` (`lib/media.ts:62`) rewrites paths to
   `{baseUrl}/api/files/download?path=…&token=…` in remote mode, and
   `gatewayMediaDataUrl` fetches via `GET /api/media`. Backend endpoints exist
   (`web_server.py:1373` `/api/media`, `:1669` `/api/files/download`, with
   `_QUERY_TOKEN_API_PATHS` allowing query-token auth for download).

### BUG-C — `mediaExternalUrl` token-less fallback returns a guaranteed-broken `file://` (audit)

`lib/media.ts:62-78`: in remote mode it requires `conn.baseUrl && conn.token`; when the
connection is **cookie/OAuth-authenticated with no `token`** (Ace's config:
`authMode: "oauth"`), it silently falls through to `file://${path}` — a client-local URL
for a gateway file. Every consumer inherits the breakage: artifacts panel `artifactHref`
(`artifacts/index.tsx:128-138` routes `/`- and `file://`-prefixed values through it),
"Open audio/video file" button (`markdown-text.tsx:136`), generated-image open
(`generated-image-result.tsx:103`). **This is very likely the exact path behind BUG-B's
artifact-panel case.** Phase 0 confirms whether Ace's live connection carries a token.

### BUG-E — review/file-tree Reveal/Rename/Delete = wrong-machine ops on GATEWAY files (audit, Fable-5 pass)

The review pane's file tree (`store/review.ts`) is populated via `desktopGit()` — which IS
remote-aware (`desktop-git.ts:100` → `/api/git/*`), so in remote mode it lists **gateway**
files. But its right-click actions route through the **local** `desktop-fs` helpers:
`revealFile`→`revealDesktopPath` ("Local only", `desktop-fs.ts:126`),
`executeFileRename`→`renameDesktopPath` ("Local only", `:131`),
`executeFileDelete`→`trashDesktopPath` ("Local only", `:144`) — all via
`store/file-actions.ts:52/82/87`, wired into `right-sidebar/review/file-tree.tsx:426`. So in
remote mode: **Reveal** silently does nothing (Finder can't show a gateway path), and
**Rename/Delete** either error or — worse — operate on a *same-named client path if one
exists* (wrong-machine mutation). This is INV-2's exact violation on a mutating surface.
**Phase-0 ground-truth (OQ-5): the assumed backend rename/trash equivalents DO NOT EXIST** —
`web_server.py` exposes only `/api/fs/{list,read-text,write-text,read-data-url,git-root,
default-cwd}`. Routing through the facade therefore requires **building new destructive
gateway endpoints** (auth + path-allowlist + repo-root binding + reversible trash + its own
review). **v1 scope (per D-11): interim guard only** — disable Rename/Delete in the
remote-mode review tree with the D-4 message, and make Reveal an explicit "not available for
a remote file" (there is no client Finder target). The facade routing + new endpoints are a
**named follow-up PR**, gated by their own review + the RC-14 hardening; NOT this PRD.

### BUG-D — audio/video in remote mode short-circuits to a LOCAL stream URL (audit)

`markdown-text.tsx:115-123`: the `mediaKind ∈ {audio,video}` check returns
`mediaStreamUrl(path)` (custom `hermes-media://stream/` scheme reading the **client**
disk) **before** the `isRemoteGateway()` branch is reached. Remote-mode audio/video
players therefore point at nonexistent client paths. (Images take the remote branch
correctly — the check order is the bug.)

### Audited and CLEAN (for the record)

- `desktop-fs.ts` facade (readDir/readFileText/readFileDataUrl/writeText/gitRoot/diff):
  remote-aware by design. ✅
- Composer image preview: fixed by #167. ✅ Submit-time image upload
  (`image.attach_bytes`): reads client-local staged file — correct by design. ✅
- `watchPreviewFile` live-reload: explicitly guarded by `isDesktopFsRemoteMode()`
  (`preview-pane.tsx:411`). ✅
- `markdown-text` / `directive-text` / `generated-image` **image** rendering: routed
  through `gatewayMediaDataUrl` in remote mode. ✅
- `revealPath` in the **projects sidebar** (`projects.ts:779`, `project-menu.tsx:168`,
  `workspace-header.tsx:124`): reveals gateway paths in client Finder — wrong remotely, but
  **low-harm** (Finder no-ops) and non-mutating. Fixed opportunistically in Phase 3 (make it
  an explicit "not available for a remote workspace" per D-4); NOT a load-bearing gate.
- `desktopGit().scanRepos` returns **`[]` in remote mode** (`desktop-git.ts:96`) — a
  **functional gap** (remote repo auto-discovery doesn't work), not a correctness/safety bug.
  Noted for a follow-up; out of scope for this PRD (no wrong-machine op, no data loss).

## 4. Resolved Decisions

- **D-1 (fork):** No new repo. `apps/desktop` inside `Kyzcreig/hermes-agent` IS our
  desktop fork; we ship fleet fixes there (auto-merge own fork when green) and upstream
  cherry-picks case-by-case. Keeps one divergence surface instead of two.
- **D-2 (reuse the existing remote seam):** All fixes route through the existing
  remote-aware helpers (`isRemoteGateway`, `gatewayMediaDataUrl`, `mediaExternalUrl`,
  `/api/media`, `/api/files/download`) — extend them where deficient (BUG-C), never mint a
  parallel resolution system. (Discarded-intermediate rule: the engine exists; stop
  discarding it.)
- **D-3 (/compress ownership):** In the desktop/TUI-gateway context, `/compress` must be
  executed **exactly once, against the live session** — the `_mirror_slash_side_effects`
  live path is the single implementation; the slash worker must NOT also run its own
  compress against its stale copy. Route `compress` around the worker (same mechanism as
  `_PENDING_INPUT_COMMANDS` direct-routing), returning the live result as the primary
  output, not a `warning`.
- **D-4 (fail loud, honestly):** Any remote-mode surface that cannot be made to work must
  return an explicit "this file lives on the gateway; <action>" message — never the OS's
  misleading "No application found" or a silent no-op.
- **D-5 (open-vs-download semantics for BUG-B — PREFERENCE ORDER, not a coin flip; RC-7):**
  For the HTML **preview pane** webview, the **required** shape is an **opaque/null-origin
  `blob:`/`data:` document built from fetched bytes** — NEVER a `<webview src>` whose origin
  equals the gateway API origin (`http://mac-studio-m3u:9119/...`). Rendering agent-generated
  HTML at the gateway's own origin lets a malicious artifact `fetch('/api/...')` same-origin on
  the desktop's ambient session auth — a stored-content→authenticated-API escalation strictly
  worse than the bug being fixed. (A sandboxed non-gateway-origin webview is the only
  alternative; default to blob/data.) For **"open in the OS/browser"** on a gateway file:
  fetch bytes via the session API → stage to a client temp file (D-7) → `shell.openPath`/open
  the temp. The authenticated-download-URL-in-the-default-browser path is **only** permitted
  when D-10's scoped-token condition holds; for Ace's token-less oauth config it's unavailable,
  so bytes→temp is the load-bearing implementation. **Stated asymmetry (pass-4 lens):** an
  externally-opened HTML temp runs in the user's browser with full JS and no CSP — same trust
  level as local mode today (user-initiated open of their own artifact), so not a regression;
  note the provenance caveat (pass-6): a remote artifact's content can be shaped by
  prompt-injected gateway output, accepted v1 because the open is user-initiated and the
  gateway is SameSite/CORS-locked. The hardened surface is the in-app preview pane, which
  renders untrusted content by default.
- **D-6 (BUG-A splits into two independently-shippable defects — RC-1/RR-1):**
  **BUG-A(a) = routing/UI contradiction** (worker+mirror double-run, 4018 fallback) — owned
  by this PRD, Phase 1. **BUG-A(b) = live compress reduces nothing at 355 msgs** — root cause
  isolated in Phase 0; **if Phase 0 finds it's a compression-*engine* bug that also affects
  auto-compaction (the §2 non-goal), it FORKS to its own PR** and is NOT gated inside this PRD.
  Acceptance splits accordingly (AC-1a routing / AC-1b reduction). A builder may NOT "fake-pass"
  token-reduction by touching auto-compaction internals under this PRD; if BUG-A(b) is engine-deep,
  this PRD ships the routing fix + an honest "compress ran once on the live session; token
  reduction tracked in <fork PR>" and stops. If Phase 0 finds BUG-A(b) is *caused by* BUG-A(a)'s
  double-execution (the concurrent-lock self-contention hypothesis), then fixing (a) fixes (b)
  and AC-1b passes here — that is the good case, but it must be *proven* live, not assumed.
- **D-7 (temp-file lifecycle for the bytes→temp fallback — RC-2/RC-10/RC-11):** staged
  gateway bytes go to a **per-user OS-temp subdir** — Electron `app.getPath('temp')` /
  Node `os.tmpdir()` (`$TMPDIR`, per-user), with the subdir `mkdir`'d **mode 0700** (not a
  world-readable literal `/tmp/...` — these bytes are sensitive gateway artifacts:
  transcripts, generated docs). Cleanup:
  - **In-app-owned surfaces** (preview webview): unlink on window close.
  - **External-app path** (`shell.openPath` → Preview/browser/player, incl. a/v
    download-to-play): **DO NOT unlink on handoff** — `shell.openPath` returns after *launch*,
    not after the app *reads* the file; unlinking on handoff races the reader and reproduces
    the exact silent-broken symptom this PRD kills. Rely on the **TTL sweep — on app start
    AND a periodic in-session sweep** (RR-8: Ace runs long-lived desktop sessions; a
    start-only sweep would let sensitive temps outlive their TTL for the whole session).
  Stale-snapshot-on-open (edit-on-gateway after open diverges) is **acceptable v1** (stated,
  not silent). Hygiene test: N opens leave 0 orphaned temps after close/TTL, AND an
  external-open temp survives long enough to be read (no handoff-unlink race).
- **D-8 (INV-2 is an ENFORCED import-boundary lint rule — RC-3/RC-9/RC-12):** a taint check
  ("is this argument remote-aware?") is undecidable in `no-restricted-syntax`, so the
  enforceable shape is an **import-boundary / file-allowlist rule**. The banned set is BOTH
  layers: (a) raw `shell.*` / `fs.*` in the renderer, AND (b) the **local-only symbols** —
  precisely named (pass-7 RC-1, these are TWO DIFFERENT symbols): the local-only `desktop-fs`
  exports (`revealDesktopPath`, `renameDesktopPath`, `trashDesktopPath` — marked "Local
  only") and the **raw bridge property `window.hermesDesktop.readFileDataUrl`** (local-only
  read). The remote-aware facade export **`readDesktopFileDataUrl`** (`desktop-fs.ts:97`) is
  NOT banned — it's the allowlisted seam. (The earlier short-name "readFileDataUrl" in this
  rule was ambiguous between the two; the lint targets the bridge property + local-only
  exports, never the facade.) — because BUG-E's defective calls are the *local-only
  exports* one hop up from `shell.*`, not raw `shell.*` (RC-12: a rule that bans only raw
  `shell.*` flags the legitimately-allowlisted `main.cjs` bridge and MISSES the renderer call
  that is the actual bug). Allowlist = the remote-aware facade + `main.cjs` (the real local
  bridge) + enumerated legitimate local-mode sites (e.g. `use-prompt-actions/utils.ts`
  submit-path reads, the #167 composer-preview local-first read). A new banned call outside
  the allowlist FAILS CI. **Escape hatch (RC-9, concretely):** set eslint `noInlineConfig:
  true` for these rules (or `reportUnusedDisableDirectives` + ban disabling this rule) so a
  per-line `// eslint-disable` cannot silently bypass INV-2 — a bare "prove the self-test
  catches a disable" is unachievable without `noInlineConfig`. **AC-8 self-test uses a
  BUG-E-shaped case:** add a `renameDesktopPath`/`trashDesktopPath` call in a remote-aware
  renderer file (NOT a `shell.openPath`) and assert the rule FAILS; **AND a false-positive
  assertion (pass-7 RC-1): the rule does NOT fire on an allowlisted facade call
  (`readDesktopFileDataUrl` from a remote-aware file)** — prove it spares good ones, not just
  that it catches bad ones. The §Audit table is documentation; this rule is the proof.
- **D-9 (Phase-0 exit contract — RC-4):** Phase 0 produces a repro log in §Review Log.
  **Any finding that contradicts §3 mechanics requires a PRD revision + a fresh review pass
  before Phase 1/2 build** — not an in-flight pivot. Apollo reviews the repro log against
  the live gateway logs; a falsified mechanism re-triggers the spec→review loop for the
  affected phase only.
- **D-10 (download token must be scoped before it touches the browser — RC-6, security):**
  D-5's browser-URL path hands a `?token=` to the **default system browser** (→ history,
  shared profile, lives as long as the token). Phase 0 confirms whether
  `/api/files/download`'s query token is **path-bound and/or short-TTL**. If it is a
  long-lived session token, the browser-URL path is **forbidden** — use bytes→temp (D-7)
  instead, or mint a scoped one-shot token. No long-lived credential may leave the app into
  the browser.
- **D-11 (BUG-E v1 = interim guard; facade routing DEFERRED — RC-13/RC-15, RESOLVED by
  Phase-0/OQ-5):** Phase 0 confirmed the mutating gateway endpoint (`/api/fs/*` rename+trash)
  **does not exist** — building it is a destructive-endpoint project with its own hardening
  (auth + path-allowlist + **same repo root the review tree listed via `/api/git/*`** — a
  root mismatch is a fresh wrong-target mutation — + reversible Trash-not-unlink) and its own
  review + the RC-14 negative tests. **That is a named follow-up PR, not this PRD.** This PRD
  ships the **interim guard** (RC-15 — the only data-loss bug must not ship last): in Phase
  1's window (or a hotfix), **disable Rename/Delete in the remote-mode review tree** and show
  the D-4 message, so there is NO window where a destructive wrong-machine op is reachable.
  **Guard completeness (OQ-6):** the guard must cover EVERY mutating entry into the review
  tree (context menu, keyboard shortcut, any drag-rename) — enumerate entry points in the
  build; a guard that misses a second entry point reopens the window it exists to close.
  **Tracking (pass-6 RC-2#2):** the follow-up is NOT left unnamed — **filing the follow-up
  issue on `Kyzcreig/hermes-agent` (facade rename/delete + `/api/fs/*` destructive endpoints
  + RC-14 hardening) is a REQUIRED closeout artifact of this PRD** (AC-10); "interim" is only
  truthful if the successor exists with an owner. The guard is a deliberate functional
  regression (remote rename/delete disabled) and must not become permanent-by-neglect.
- **D-12 (HTML preview defense-in-depth — RC-16):** the opaque-origin blob (D-5) blocks
  *same-origin* `fetch('/api/...')` but NOT a null-origin doc issuing *cross-origin*
  cookie-bearing requests if the gateway oauth cookie is `SameSite=None` (plausible for a
  two-host thin-client↔gateway setup). So the preview webview ALSO renders with **JavaScript
  disabled and/or a restrictive CSP** (`default-src 'none'; img-src data: blob:`), OR Phase 0
  confirms the gateway cookie is `SameSite=Lax/Strict` + CORS-locked. AC-2's origin assertion
  is concrete: evaluate `location.origin` inside the preview, assert opaque/`null`.

## 5. Implementation Phases

### Phase 0 — Live reproduction + premise isolation (no code changes) — GATE
Ship: a written repro log in this doc's §Review Log. **Exit contract per D-9: any finding
that contradicts §3 requires a PRD revision + fresh review pass before build.**
- Reproduce BUG-A live on Ace's MBP↔Studio pair; capture the exact RPC sequence
  (`slash.exec` result vs thrown; whether `command.dispatch` fires on first or second
  invocation) from gateway logs.
- Isolate why the LIVE mirror compress reported "No changes" at 355 msgs (BUG-A(b)):
  instrument `_compress_session_history` / the concurrent-lock path (temporary logging),
  capture the summarizer call result AND whether the concurrent-compression lock returned
  `messages` unchanged. Record the **resolved** `compression.enabled` value (D-6 / config-drift
  lens), not just the code path. **Classify BUG-A(b): routing-caused (fixed by Phase 1) vs
  engine-deep (forks per D-6).**
- Confirm whether Ace's oauth connection carries a query token (`conn.token`) — **already
  ground-truthed: it does NOT** (`connection.json` has only `url`+`authMode:oauth`, no token).
  So D-5's browser-URL path is **unavailable for Ace's config**; the bytes→temp path (D-7) is
  the load-bearing one and MUST get live E2E (RR-2). Still confirm the token's scope/TTL (D-10)
  in case a token appears for other users.
- **BUG-E backend dependency (OQ-5, RC-13):** confirm `/api/fs/*` exists with **rename AND
  trash/delete**, is authenticated + path-allowlisted, resolves against the **same repo root**
  the review tree lists via `/api/git/*`, and whether delete is reversible (Trash vs `unlink`).
  If absent/read-only, Phase 3 grows a destructive-endpoint build (re-scope via D-9).
- **HTML-preview cookie posture (RC-16):** confirm whether the gateway oauth cookie is
  `SameSite=Lax/Strict` + CORS-locked (if so, D-12's CSP is belt-and-suspenders; if
  `SameSite=None`, the CSP/JS-disabled webview is mandatory).
- *Verification:* repro log cites gateway log lines + the isolated cause + the BUG-A(b)
  classification. **Any temporary instrumentation added for the repro is reverted before any
  ship (pass-5 RC-4).** **Gate: Phases 1-2 designs are confirmed/corrected against this before
  build (D-9). The BUG-E interim guard is DECOUPLED from this gate** — OQ-5 is already
  resolved; the guard can ship as an immediate hotfix without waiting for the BUG-A live repro
  (pass-5 DevOps lens).

### Phase 1 — `/compress` routing correctness (BUG-A(a)) [+ BUG-A(b) only if routing-caused]
Ship: `tui_gateway/server.py` + `apps/desktop/src/.../slash.ts` changes.
- Route `compress` past the slash worker (direct-route set, same mechanism as
  `_PENDING_INPUT_COMMANDS`), execute once on the live session, return the live feedback as
  the primary output (not a `warning`).
- **No-op write guard (Phase-0 R3 — REQUIRED):** `_compress_session_history` must NOT swap
  `session["history"]` / bump `history_version` when compression returned the input unchanged.
  **Predicate (pass-9 RC-2, pinned): object identity — `compressed is before_messages`** (the
  lock-skip and the engine's no-op paths return the same list object). NOT `removed == 0`,
  which would also suppress a legitimate same-length rebuild. Unit tests BOTH directions
  (RC-2): (a) lock-skip no-op → `history_version` untouched, history object unswapped; (b) a
  genuine compaction → version DOES bump and history IS swapped (over-fire = silent dropped
  compaction, the worse failure).
  **Class scope (pass-9 RC-1 — ground-truthed from source, 2026-07-02):** the poisoning class
  is **manual+manual only**. Auto-compaction never writes `session["history"]` mid-turn — it
  mutates `run_conversation()`'s in-loop `messages`; the only session write-back is the
  turn-end block (`server.py:8646-8651`), which is itself version-guarded (mismatch → output
  NOT written, warning surfaced). And manual/auto cannot overlap: manual `/compress` is
  rejected while `session.running` (`_MUTATES_WHILE_RUNNING`, `:12434`), and auto-compaction
  only fires inside a running turn. So the R3 site fix covers the class; residual (stated):
  a turn *started* during a slow manual compress supersedes it via the version check — the
  compress result is dropped in favor of the newer turn, which is the correct, existing,
  deliberate behavior ("avoid session fork").
  **Acceptance (pass-9 RC-3 — prove the effect, not the proxy):** E2E on the live pair:
  double-press `/compress` inside the real compaction window (~38-74s) → the FIRST press's
  real compaction **lands** (message count + tokens DROP; one coherent report), the second
  returns an honest "already compressing" — not "version untouched" as a proxy. This is the
  exact R3 repro condition, now required to pass.
- **Version-bump consumers (pass-9 residual):** before suppressing the no-op bump, grep
  `history_version` consumers to confirm none treats a bump as a "compress happened" signal
  (UI spinner/refresh) — the bump must be a write-fence only.
- `command.dispatch` gains an honest `compress` case (or the desktop stops falling back
  for it) so the `4018` contradiction cannot recur.
- If Phase 0 proved BUG-A(b) is routing-caused (double-execution → concurrent-lock
  self-contention), the single-execution fix resolves it; prove reduction live (AC-1b here).
  If Phase 0 proved BUG-A(b) is engine-deep, it FORKS (D-6) — this phase ships routing only.
- **Command-class audit (RC-8) — now MEASURED (Phase-0 R7):** the sweep found the stale-worker
  class live: `/usage`, `/history`, `/prompt` return worker-stale nonsense; `/status` shows the
  worker's Tokens:0. Disposition: serve session-READING commands from the LIVE session (same
  seam as compress; read-only = lower risk). Also from R7: `/model` returns empty output (echo
  the current model), `/clear` dumps the CLI splash banner into the desktop chat (suppress for
  desktop clients), `/models`//`/rename`//`/effort` → alias or a helpful error. Enumerate the
  remaining `_mirror_slash_side_effects` × worker set (`/undo`, `/steer`, `/snapshot`, `/learn`,
  …) as already-safe / fixed-here / named-follow-up. Don't leave the class unexamined behind
  INV-1.
- **BUG-E interim guard (RC-15 — ship in this phase's window):** disable Rename/Delete in the
  remote-mode review tree + show the D-4 "not available for a remote file" message, so the only
  data-loss footgun is closed immediately. **The guard persists for the life of this PRD —
  Phase 3 keeps it in place** (Phase 3 ships no routing; per D-11/OQ-5 the facade routing was
  deferred). It is lifted ONLY when the **named follow-up PR's** facade + endpoints land and
  pass their own gates — never in Phase 3.
- **Config-drift convergence (config-drift lens):** the compress call-sites have divergent
  `force=` semantics — converge the TUI-gateway path to CLI/manual intent, don't patch only
  one. **Reconciled scope table (pass-6 RC-2#1 — the changed site, its containing function,
  and reachability):**

  | Call site | Function containing it | Current `force=` | Target | Reachable by auto-compaction? |
  |---|---|---|---|---|
  | `tui_gateway/server.py:2855` (`agent._compress_context(...)`) | **INSIDE `_compress_session_history`** (def `:2820`; the `:2853` cited earlier is its comment line) | absent (False) | **True** | **NO** — `_compress_session_history` has exactly two callers: `session.compress` RPC (`:7619`) and `_mirror_slash_side_effects` (`:12472`), both manual `/compress` |
  | `cli.py:9311/:9316` | CLI `_manual_compress` | True | unchanged | NO — manual only |
  | `gateway/slash_commands.py:3119` | gateway manual `/compress` | True | unchanged | NO — manual only |
  | `agent/conversation_loop.py:3013/:3259/:3482/:4614`, `agent/turn_context.py:403` | **auto-compaction** inside `run_conversation()` | default False | **unchanged — NOT touched** | YES (they ARE auto-compaction) |
  | `gateway/run.py:10601` (session hygiene), `acp_adapter/server.py:1935` | other consumers | as-is | unchanged | out of scope |

  The ONLY modified line is `server.py:2855`, and its containing function is reachable
  exclusively from the two manual paths — the change **cannot** alter auto-compaction
  behavior; INV-4's carve-out is proven in-doc, not deferred to a build-time grep.
- *Unit:* worker-bypass routing test; dispatch-case test; single-execution assertion;
  **force=-behavior test (pass-6 RC-2#3):** assert the manual compress path passes
  `force=True` through to the compressor (cooldown cleared / `_clear_compression_failure_cooldown`
  invoked) AND that the auto-compaction path still calls with default `force=False` — the
  deliberate INV-4 carve-out gets its own gate, not just "suite green".
- *E2E (AC-1a, always):* remote session >100 msgs → `/compress` yields ONE coherent report,
  zero `4018`, exactly one compress marker in the gateway log.
- *E2E (AC-1b, if in-scope):* message count + token estimate DROP. If forked, this is the
  fork PR's gate, not this one's.
- *Negative:* `/compress` while a turn is running → single honest "session busy" line, no
  4018, no double execution. **Stale-worker (RR-3):** confirm the bypassed-worker path leaves
  no stale slash-worker state that leaks into the next `/compress`.

### Phase 2 — Remote file open + HTML preview (BUG-B + BUG-C)
Ship: `electron/main.cjs`, `lib/media.ts`, `preview-row.tsx`, `preview-pane.tsx`,
`local-preview.ts` (+ backend only if a scoped-token endpoint is needed per D-10).
**Build ordering (RC-5): the URL-token path may NOT be coded until Phase 0 answers OQ-1/D-10;
for Ace's token-less config the bytes→temp path (D-7) is the primary implementation.**
- `mediaExternalUrl`: eliminate the broken `file://` fallback in remote mode (BUG-C) —
  a working authenticated URL (scoped token only) or an explicit error per D-4; never a
  silent client-local `file://`.
- Preview-row "open in browser" + artifacts-panel open: remote-aware per D-5; bytes→temp
  with the D-7 lifecycle for the token-less path.
- HTML preview pane webview: remote-aware source per D-5 (authenticated URL or data/blob).
- *Unit:* `mediaExternalUrl` remote/no-token case (asserts NO `file://`); target-URL builders;
  temp-file GC (D-7 hygiene: N opens → 0 orphans after close/TTL).
- *E2E:* on the live pair (token-less oauth = Ace's real config), click a gateway-side `.html`
  artifact → renders in browser and/or preview pane; fetched content hash matches the gateway
  file. An image and a text file open likewise. **Live E2E covers the bytes→temp branch —
  the ONLY branch reachable on the live pair (token-less oauth, per Phase 0).
  DECISION (pass-5 RC-3): the URL-token branch is NOT BUILT in this PRD** — no runtime
  selection path is wired for it (YAGNI: no fleet config can exercise it, and dead code
  without E2E validation is against house rules). D-10 remains the design constraint the
  **follow-up PR** must satisfy if/when a scoped-token config exists. `mediaExternalUrl` in
  remote mode resolves to the bytes→temp/data-URL path or an explicit D-4 error — nothing
  else.
- **Observability (pass-7 lens):** a low-noise debug log line on every remote-mode fallback
  decision (bytes→temp taken / honest-error shown / TTL sweep ran, with path + reason) — the
  original bugs were only found when Ace hit them; this shortens the next diagnosis. Local
  logging only, no telemetry.
- *Negative:* gateway file deleted after listing → explicit not-found toast (no OS error);
  path-traversal (`?path=../../etc/passwd`) rejected by backend hardening (assert 403/400);
  **token-scope (D-10): confirm no long-lived token is written to browser history.**

### Phase 3 — Audit sweep fixes + enforced class guard (BUG-D + BUG-E guard + INV-2)
Ship: `markdown-text.tsx` ordering fix; **BUG-E interim guard** (D-11); the **enforced
lint/CI rule (D-8)**; the §Audit table as documentation.
- Reorder the audio/video branch after the remote check (`markdown-text.tsx:115`); remote a/v
  uses the authenticated download/stream URL (or explicit "not streamable remotely" per D-4 if
  Range support is missing — **OQ-3 is owned HERE, a Phase-3 scope decision**; rec:
  download-to-play v1 via bytes→temp/D-7). Temp a/v obeys D-7.
- **BUG-E (v1 = guard only, per D-11/Phase-0):** disable Rename/Delete in the remote-mode
  review tree across **every** mutating entry point (context menu, keyboard shortcut, any
  drag-rename — OQ-6), showing the D-4 message; make Reveal an explicit "not available for a
  remote file" — never a silent no-op or a same-named-client-path mutation. Opportunistically
  fix the projects-sidebar `revealPath` the same way. **The facade rename/delete + new
  `/api/fs/*` destructive endpoints + RC-14 hardening + SSH-verified mutation assertions are
  the named follow-up PR's scope and gates, not this PRD's.**
- **Add the D-8 lint rule** (import-boundary/file-allowlist) that fails CI on a new bare
  `shell.*`/`fs.*`/`window.hermesDesktop.readFileDataUrl`/local-only-export call outside the
  allowlisted facade (per D-8's precise symbol split — the facade's `readDesktopFileDataUrl`
  is allowlisted, not banned). This is INV-2's real proof — and it would have caught BUG-E at
  authoring time.
- **Periodic in-session TTL sweeper (RR-8/pass-7 RC-3 — explicit build item, not implied):**
  ship the D-7 temp-sweeper as a named deliverable: runs on app start AND on a periodic
  in-session timer; covered by the AC-7 hygiene test.
- *Unit:* mediaSrc resolution matrix (local/remote × image/audio/video/html); **lint-rule
  self-test (a deliberately-bad new call site FAILS; `// eslint-disable` cannot bypass);**
  BUG-E guard test (remote rename/delete are blocked at every entry point; local mode
  unaffected).
- *E2E:* remote session: play an agent-generated audio file (or honest fallback); attempt
  rename & delete from the review tree → blocked with the D-4 message, gateway AND client
  filesystems untouched (verified over SSH + locally).
- *Negative:* remote Reveal on a gateway file → explicit "not available" message, no silent
  no-op; the guard covers context menu + keyboard + drag entry points (OQ-6 enumeration).

## 6. Constitution / Invariants

- **INV-1 (single execution — scoped to `compress`; RC-8):** The `/compress` command
  executes at most once per user invocation, on the live session. *Proof:* Phase-1 E2E +
  a log-grep asserting one compress marker per invocation, AND a unit test asserting the
  *routing* (worker bypassed) not just output shape. **Scope note:** the worker-vs-mirror
  double-execution is structural and may affect other mirrored session-mutating commands;
  INV-1 is deliberately narrowed to `compress` (the reported bug). Phase 1 audits the
  `_mirror_slash_side_effects` set × the worker path and **dispositions every other
  session-mutating command** (fix-in-scope or file as a named follow-up) — we do NOT ship a
  class-wide invariant on a one-command proof (the "fix the class, not the site" rubric).
- **INV-2 (no wrong-machine file ops):** No renderer/electron path may hand a
  gateway-side path to a client-local `shell.*`/`fs` API in remote mode. *Proof:* the
  **enforced D-8 lint/CI rule** (fails on a new bare call site) — NOT the doc table alone.
  The Phase-3 §Audit table documents current dispositions; the lint rule enforces them.
- **INV-3 (honest failure):** remote-mode failures name the real cause (gateway file,
  auth, endpoint) — asserted in Phase-2 negative tests.
- **INV-4 (existing local behavior byte-identical — ONE stated carve-out):** every touched
  surface behaves identically in local mode, **except** the deliberate `force=` convergence
  (Phase 1): the TUI-gateway compress path's prior no-force call (`server.py:2855`, inside
  `_compress_session_history` — Phase-1 scope table) was the
  latent defect — local desktop `/compress` gains the same cooldown-bypass the CLI's manual
  compress has had since #15281 (`cli.py:9316`). That is the intended fix, stated here so
  INV-4 doesn't assert byte-identical while deliberately changing the byte. **By-design note
  (pass-7):** the cooldown-bypass is manual-only — a user spamming `/compress` after a
  summarizer failure re-hammers the summarizer each press; accepted as CLI parity (#15281),
  auto-compaction keeps the cooldown. *Proof:* existing
  local-mode tests stay green; no OTHER default changes.
- **INV-5 (auth):** no new unauthenticated file access; download/media endpoints keep
  their existing hardening (path allowlist, token/query-token). Negative test in Phase 2.

## 7. Risks & Mitigations

- **R1: Phase-0 falsifies a mechanism** (e.g. the 4018 trigger, or "No changes" is a
  config issue not a code bug). Mitigation: Phase 0 is a hard gate; designs corrected
  before build (this is why it exists).
- **R2: token-less oauth connections** make D-5's URL path impossible without backend
  work. Mitigation: Phase 0 decides; fallback design (bytes→temp file) needs no backend
  change.
- **R3: double-compression race** if both auto-compaction and manual compress fire.
  Mitigation: the live path already serializes via `history_lock`; Phase-1 E2E covers a
  busy-session invocation.
- **R4: upstream drift** — fixes land on the fork while upstream diverges. Mitigation:
  same cherry-pick flow as #56827; not a gate.
- **R5: rollback / ship mechanism (DevOps lens).** Each phase ships as its own fork PR,
  auto-merged when green (D-1), then the desktop app is rebuilt+installed via
  `desktop-update.sh` (same as #167). **Rollback:** revert the phase's squash commit on
  `fork/main` + rebuild, OR restore the prior app from `/Applications/.hermes-app-backups/`
  (staged by every `desktop-update.sh --apply`). The `/compress` routing change (Phase 1)
  is the one with local-mode regression risk — INV-4 + the existing desktop suite gate it,
  and its rollback is a single-commit revert.

## 8. Acceptance Criteria

- [ ] AC-1a (routing, this PRD): `/compress` in a remote desktop session yields ONE
  coherent report and zero `4018`, with exactly one compress marker in the gateway log per
  invocation. Evidence: Phase-1 E2E transcript + `grep`-count of compress markers.
- [ ] AC-1b (token reduction): a remote `/compress` on a >100-message session reduces
  message count + token estimate. **Owned here ONLY if Phase 0 proved BUG-A(b) is
  routing-caused; else this AC moves to the forked engine-fix PR (D-6).** Evidence: before/after
  counts from the live run (this PRD) or the fork PR's gate.
- [ ] AC-2: clicking a gateway-side HTML artifact in remote mode renders it (preview pane
  via **opaque-origin blob/data**, never gateway-origin; and/or OS open via bytes→temp).
  Evidence: Phase-2 E2E screencap + fetched content hash matches gateway file + a check that
  the webview document origin is NOT the gateway API origin (RC-7).
- [ ] AC-3: `mediaExternalUrl` never returns a `file://` URL in remote mode. Evidence:
  unit matrix + grep.
- [ ] AC-4: remote a/v media resolves through the remote path (or an explicit honest
  message). Evidence: Phase-3 unit matrix + live check.
- [ ] AC-5: call-site audit table committed with every fs-touching bridge call
  dispositioned. Evidence: the table in this doc + grep parity check.
- [ ] AC-6: all touched local-mode behavior unchanged (existing desktop test suite green;
  typecheck/eslint clean). Evidence: CI green + local-mode E2E unchanged.
- [ ] AC-7 (D-7 hygiene): N remote "open" operations leave 0 orphaned temp files after
  window-close/TTL. Evidence: temp-GC unit test + a live count.
- [ ] AC-8 (D-8 enforced INV-2): the lint/CI rule FAILS on a deliberately-added banned call
  (BUG-E-shaped local-only export in a remote-aware file, and a bare `shell.openPath`/`fs.*`)
  AND does NOT fire on an allowlisted facade call (`readDesktopFileDataUrl` — pass-7
  false-positive case). Evidence: both self-tests in CI.
- [ ] AC-9 (BUG-E, v1 guard): in remote mode, review-tree Rename/Delete are **disabled at
  every entry point** (context menu, keyboard, drag — OQ-6) with the D-4 message; gateway AND
  client filesystems untouched (verified over SSH + locally); Reveal shows an explicit "not
  available" message. Evidence: Phase-3 E2E + guard unit test. **The SSH-verified
  gateway-mutation assertion belongs to the follow-up facade PR, not this PRD.**
- [ ] AC-10 (follow-up filed — pass-6 RC-2#2): the BUG-E facade/destructive-endpoint
  follow-up issue exists on `Kyzcreig/hermes-agent` (title, scope = facade rename/delete +
  `/api/fs/*` endpoints + RC-14 hardening + D-10 token design, owner Apollo) BEFORE this
  PRD closes. Evidence: issue URL in the closeout table.
- [ ] AC-11 (R7 slash sweep — pass-9 RC-4): re-run the Phase-0 R7 sweep harness after Phase 1:
  `/usage`, `/history`, `/prompt` return LIVE-session data (no "(._.)" stale-worker output),
  `/status` reports the live token count, `/model` echoes the current model, `/clear` does not
  dump the CLI splash banner. Evidence: sweep table before/after in the closeout.
- [ ] AC-12 (R3 harm closed — pass-9 RC-3): double-press `/compress` inside the live
  compaction window → the first press's real compaction LANDS (count+tokens drop), second gets
  an honest "already compressing" line. Evidence: Phase-1 E2E transcript + agent.log excerpt.

## Review Log

### Pass 1 — Opus (claude-api-proxy), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.2
No hard blockers. 6 Required Changes, all folded:
- **RC-1 (conflated defects):** split BUG-A into (a) routing + (b) token-reduction; added
  **D-6** (fork BUG-A(b) if engine-deep), split **AC-1 → AC-1a/AC-1b**, phase text updated.
- **RC-2 (temp lifecycle):** added **D-7** (app-scoped temp dir, unlink on close + TTL sweep,
  stale-snapshot acceptable-and-stated), **AC-7** hygiene test.
- **RC-3 (grep-table rots):** **INV-2 upgraded to an enforced lint/CI rule (D-8)**; table
  demoted to documentation; **AC-8** lint self-test.
- **RC-4 (Phase-0 exit contract):** added **D-9** (falsified mechanism → PRD revision + fresh
  review pass, Apollo reviews the repro log).
- **RC-5 (ordering):** Phase-2 URL-token path build **BLOCKED until Phase 0 answers OQ-1/D-10**
  (stated in Phase 0 + Phase 2).
- **RC-6 (token→browser exfil):** added **D-10** (download token must be path-bound/short-TTL
  before touching the browser; else forbidden → bytes→temp). Ground-truthed: **Ace's oauth
  connection has NO token**, so the browser-URL path is unavailable for his config and
  bytes→temp (D-7) is load-bearing — folded into Phase 0/2 + RR-2.
- Also: added **R5 rollback/ship** note (DevOps lens), and captured the **live OQ-2 clue**
  (concurrent-compression lock returns messages unchanged — plausible self-inflicted "No
  changes" from BUG-A's double-execution) into §3 BUG-A(b) as a Phase-0 hypothesis (not a
  settled cause).

### Pass 2 — Opus (claude-api-proxy), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.3
Confirmed pass-1's 6 RCs are genuinely folded (not rubber-stamped). 5 new RCs from the fold,
all folded:
- **RC-7 (webview same-origin escalation, security):** **D-5 rewritten** — HTML preview pane
  MUST use an opaque/null-origin `blob:`/`data:` document, NEVER a `<webview src>` at the
  gateway API origin (would let a malicious artifact `fetch('/api/...')` same-origin on ambient
  auth). AC-2 now asserts the origin check.
- **RC-8 (INV-1 overclaim — fix-the-class):** **INV-1 narrowed to `compress`**; added a Phase-1
  **command-class audit** dispositioning every other mirrored session-mutating command. No
  class-wide invariant on a one-command proof.
- **RC-9 (lint predicate undecidable):** **D-8 rewritten** to an enforceable **import-boundary /
  file-allowlist** rule (ban raw `shell.*`/`fs.*` outside allowlisted facade files) + AC-8 must
  prove `// eslint-disable` can't bypass it.
- **RC-10 (unlink races the reader):** **D-7** now: external-app opens (incl. a/v) rely on the
  **TTL sweep**, NOT handoff-unlink (which races `shell.openPath`'s launch-not-read return);
  only in-app-owned surfaces unlink on close.
- **RC-11 (temp perms):** **D-7** pins the temp root to per-user `app.getPath('temp')`/`$TMPDIR`
  with a **0700** subdir (not world-readable `/tmp`).
- Also folded: config-drift convergence of the 3 divergent `force=` compress call-sites into
  Phase 1; noted the live-pair E2E ACs (AC-1a/2/4/7) are **manual, non-CI** gates Apollo runs
  and records here (AC-6 "CI green" does not cover them).

**Convergence:** pass 2 returned APPROVE-WITH-CHANGES with all changes foldable in-spec and no
open blockers — converged. Remaining items are Phase-0-gated open questions (OQ-1..4), which is
correct: they need live ground-truth, not more spec iteration.

### Fable-5 deep audit — 2026-07-02 (widened sweep, model=claude-fable-5)
Second, wider pass over ALL electron IPC handlers + git cockpit + terminal + file-ops (not just
`window.hermesDesktop?.<fs>` call sites). New finding:
- **BUG-E (added):** the review file-tree is remote-aware (lists gateway files) but its
  Reveal/Rename/Delete route through the LOCAL `desktop-fs` reveal/rename/trash — a
  wrong-machine mutation on a gateway file. Added to §3 + Phase 3 + AC-9.
Confirmed CLEAN on wider sweep: the **terminal** panel spawns a local `node-pty` but is a
*client-local* terminal by design (not claimed to be the gateway shell) — not a remote-mode
bug; the **git cockpit** (worktree/branch/commit/review) is fully remote-aware via `/api/git/*`
(`desktop-git.ts`); **image-save** (`saveImageFromUrl`) handles data:/http/file so it works in
remote mode when `src` is a `data:` URL (chat images via `gatewayMediaDataUrl`) — only breaks if
fed a BUG-C `file://`, so it's covered by the BUG-C fix, not independent.
Functional gaps (noted, out of scope, no safety/data risk): `scanRepos` returns `[]` remotely;
projects-sidebar `revealPath` no-ops on gateway paths (fixed opportunistically in Phase 3).

### Pass 3 — Opus (claude-api-proxy), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.5 (CONVERGED)
Verified pass-2's RC-7..11 folds line-by-line (genuine). 5 new RCs, all about BUG-E slipping the
nets that were written before it existed — all folded, no redesign:
- **RC-12 (D-8 misses BUG-E):** BUG-E's bad calls are the local-only `desktop-fs` *exports*
  (`renameDesktopPath`/`trashDesktopPath`/…), not raw `shell.*` — so D-8 as specced would've
  missed the very bug it advertised. **D-8 banned-set now includes the local-only exports**;
  AC-8 self-test uses a BUG-E-shaped case; `noInlineConfig` closes the eslint-disable hatch.
- **RC-13 (unproven `/api/fs/*` premise):** added **D-11** + Phase-0 OQ-5 — confirm the mutating
  endpoint exists/authz/path-root BEFORE Phase 3; if absent, it's a destructive-endpoint build.
- **RC-14 (destructive op under-tested):** Phase-3 negatives now require the mutating endpoint to
  carry ≥ the read path's hardening (traversal + authz + SSH root-match).
- **RC-15 (data-loss bug ships last):** added an **interim guard in Phase 1** disabling remote
  Rename/Delete immediately, so no window exists where a destructive wrong-machine op is reachable.
- **RC-16 (SameSite cookie):** added **D-12** — preview webview also gets JS-disabled/CSP, or
  Phase 0 confirms `SameSite=Lax/Strict`; AC-2 origin assertion made concrete.

**Convergence:** pass 3 explicitly states these are "foldable in-spec, no redesign required —
fold + one confirmation pass gets this to a super-pass." All folded. The remaining items (OQ-1,
OQ-2, OQ-5) are correctly **Phase-0 live-ground-truth gates**, not spec iteration — the spec is
converged; Phase 0 answers them with the real system before the dependent phases build.

### Pass 4 — Opus (claude-bpp), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.6
Verified RC-12..16 genuinely folded. Caught the one real process failure of the day: **the
static Phase-0 findings were recorded in the addendum but NOT propagated into the body** —
leaving §3/Phase-3/D-11/AC-9 describing the falsified `/api/fs/*` path and AC-9 asserting a
behavior v1 doesn't ship (a fake gate), plus an unachievable "both branches live E2E" claim.
All four Required Changes folded:
- **CB-1 (stale body):** §3 BUG-E, D-11, Phase 3, and AC-9 rewritten to the v1-guard scope;
  facade+endpoints+RC-14 hardening explicitly the follow-up PR's gates. AC-9 now asserts what
  v1 ships (rename/delete disabled at every entry point, both filesystems untouched).
- **CB-2 (untestable branch):** Phase-2 E2E claim corrected — bytes→temp is the only
  live-testable branch on the token-less pair; URL-token branch = unit-only, dormant, no
  live-E2E claim.
- **RC 3 (INV-4 ↔ force= tension):** INV-4 now carries the ONE stated carve-out (the no-force
  call was the latent defect; local gains the CLI's cooldown-bypass intentionally).
- **RC 4 (line drift + config capture):** lock citations pinned (`:485` sid-key, `:512-556`
  acquire/skip — same lock); live repro must record resolved `compression.enabled`.
- Also folded: OQ-6 (guard entry-point enumeration) added to D-11/Phase 3/§9; RR-8 (periodic
  in-session TTL sweep, not start-only) into D-7; OQ-3 owner pinned to Phase 3; RR-9 noted.

### Pass 5 — Opus (claude-bpp), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.7
Verified pass-4's CB-1/CB-2/RC-3/RC-4 genuinely folded. One residual of the same class + three
tightenings, all folded:
- **CB-1 (last stale forward-reference):** Phase 1's guard bullet still said "removed when
  Phase 3's routing lands" — but Phase 3 ships NO routing (deferred to the follow-up PR); a
  builder following it would strip the data-loss guard prematurely. **Fixed: the guard persists
  for the life of this PRD; lifted only when the follow-up PR's facade+endpoints land.**
- **RC-2 (`force=` scope premise):** ground-truthed statically — `_compress_session_history`
  has exactly two callers (`session.compress` RPC `:7619`, mirror `:12472`), both manual;
  TUI-gateway auto-compaction fires inside `run_conversation()` (`:8672` = the gateway-side
  post-compress session-key note; the authoritative auto-compaction call sites are the
  pass-6 table's `conversation_loop.py:3013/:3259/:3482/:4614` + `turn_context.py:403`) and
  never passes
  through it → `force=True` cannot reach auto-compaction; INV-4 carve-out valid. Recorded in
  Phase 1 with a build-time grep re-verify.
- **RC-3 (dormant branch decision):** **URL-token branch NOT BUILT** in this PRD (YAGNI — no
  fleet config can exercise it; dead code without E2E is against house rules). D-10 becomes the
  follow-up PR's design constraint.
- **RC-4:** lock citation pinned once (sid-key `:485` / acquire `:512` / skip `:530-556`);
  Phase-0 instrumentation revert requirement added; BUG-E guard decoupled from the BUG-A
  live-repro gate so it can hotfix immediately.

### Pass 6 — Opus (claude-bpp), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.8
**Zero open blockers** (pass-5 CB-1 verified genuinely dead across all five sections). One
honest catch: pass-5's RC-2 "scope proof" cited `_compress_session_history`'s callers while
naming `_compress_context:2853` as the changed site — the reviewer couldn't see they're the
same code path (`:2855` is INSIDE `_compress_session_history`, `:2853` is its comment line).
All four RCs folded:
- **RC-2#1:** replaced the prose proof with a **reconciled scope table** — every
  `_compress_context` caller in the repo (11 sites), the changed line's containing function,
  and per-row auto-compaction reachability. Only `server.py:2855` changes; its function is
  reachable solely from the two manual paths. Proven in-doc; the build-grep hedge removed.
- **RC-2#2:** filing the BUG-E follow-up issue is now a REQUIRED closeout artifact (**AC-10**)
  — the interim guard may not become permanent-by-neglect.
- **RC-2#3:** added a dedicated **force=-behavior unit test** (manual path forces
  cooldown-clear; auto path stays `force=False`) — the INV-4 carve-out gets its own gate.
- **RC-2#4:** last lock sub-range citation pinned in the Phase-0 addendum (matches §3).
- Lens residual folded: D-5's external-open caveat now names prompt-injected gateway output
  as the provenance risk (acceptable v1: user-initiated + SameSite/CORS-locked).

### Pass 7 — Opus (claude-bpp), 2026-07-02 — APPROVE WITH CHANGES → folded to v0.9
**Zero open blockers** (CB-1 traced dead across all five sections; pass-6 RC-2#1..4 verified
genuinely folded). One real catch in the enforcement layer + two cleanups, all folded:
- **RC-1 (`readFileDataUrl` double-classified — enforcement integrity):** §3 listed it as a
  remote-aware facade export while D-8 banned the same short name — with `noInlineConfig`,
  either every legitimate facade read false-positives or the audit was wrong. **Resolved: they
  are two different symbols** — the banned one is the raw bridge property
  `window.hermesDesktop.readFileDataUrl` (+ the "Local only" `desktop-fs` exports); the
  allowlisted facade export is `readDesktopFileDataUrl` (`desktop-fs.ts:97`). D-8 rewritten
  with the precise split; **AC-8 gains the false-positive assertion** (rule spares the facade,
  not just catches the bad case).
- **RC-2 (citation residue):** the two remaining `:2853` cites (§3 BUG-A pt-4, INV-4) updated
  to `:2855`-inside-`_compress_session_history`; the pass-5 log's `:8672` note reconciled to
  the pass-6 table's authoritative auto-compaction call sites.
- **RC-3 (sweeper is a named deliverable):** the periodic in-session TTL sweeper is now an
  explicit Phase-3 ship item (start + in-session timer), covered by AC-7 — not "implied."
- Lens residuals folded: INV-4 note that manual cooldown-bypass is by-design manual-only
  (repeated `/compress` after a summarizer failure re-hammers the summarizer — accepted CLI
  parity); a low-noise debug log on remote-mode fallbacks (bytes→temp taken / honest-error
  shown / sweep runs) added to Phase 2 as an observability line.

### Pass 8 — Opus (claude-api-proxy), 2026-07-02 — **APPROVE (clean) → v1.0 SUPER-PASS**
Traced pass-7's three RCs against the document (not the changelog): the two-symbol
`readFileDataUrl` split holds under scrutiny (bridge property banned / facade export
allowlisted / AC-8 spares-good assertion present); citation drift dead; sweeper is a named
deliverable. **Zero blockers, zero required changes.** One optional non-gating nit (the
`:12450` def-vs-call ambiguity) — folded anyway (def `:12418` / branch `:12450` / call
`:12472`). Verdict: "The spec is converged and internally consistent. Everything remaining
is a live-execution gate, not a spec defect — correctly parked in Phase 0/D-9."

**Review loop CLOSED at 8 passes. Next: Phase-0 live repro (MBP↔Studio) through the D-9
gate, then build Phases 1→3.**

### Pass 9 — Opus (claude-bpp), 2026-07-02 — APPROVE WITH CHANGES (D-9 delta on R3+R7) → folded to v1.2
Correctly refused to stamp the post-convergence delta unexamined. 4 RCs, all folded:
- **RC-1 (is the poisoning a class?):** ground-truthed from source — **manual+manual only**.
  Auto-compaction mutates the in-loop `messages` and never writes `session["history"]` mid-turn;
  the turn-end write-back (`server.py:8646-8651`) is itself version-guarded; manual `/compress`
  is rejected while `session.running` (`_MUTATES_WHILE_RUNNING`), so manual+auto cannot overlap.
  The R3 site fix covers the class; the turn-supersedes-compress residual is stated as the
  correct existing behavior. The "escalate to blocker" branch is falsified.
- **RC-2 (both-direction gate + exact predicate):** predicate pinned to **object identity**
  (`compressed is before_messages`; `removed == 0` explicitly rejected — would suppress
  legitimate same-length rebuilds); unit tests now assert BOTH no-bump-on-noop AND
  bump-on-real-compaction (spare-the-good).
- **RC-3 (prove the effect):** added **AC-12** — the exact live-repro'd condition
  (double-press inside the 38-74s window) must show the FIRST compaction LANDING, not a
  version-counter proxy.
- **RC-4 (R7 unmeasured):** added **AC-11** — re-run the R7 sweep harness post-Phase-1; the
  stale-worker trio + `/status` tokens + `/model` echo + `/clear` banner all gated. This entry
  also records that the pass-9 D-9 scope covered BOTH R3 and R7.
- Residual folded: version-bump-consumer grep required before suppressing the no-op bump.

**D-9 delta review complete → BUILD UNBLOCKED for Phases 1-3.**

## 9. Open Questions

- OQ-1: Does Ace's oauth connection expose a query token for `/api/files/download`?
  (Phase 0; decides BUG-C fix shape.)
- OQ-2: Why did the live mirror compress report "No changes" at 355 msgs / 430K tokens —
  code bug, swallowed summarizer error, or config? (Phase 0; may spawn its own fix PR.)
- OQ-3: Should `/api/media` grow Range/streaming support for remote a/v, or is "download
  to play" acceptable v1? (Phase 3 scope decision — recommend download-to-play v1.)
- OQ-5: ~~Does `/api/fs/*` exist with rename+trash…~~ **RESOLVED (Phase 0): NO — endpoints
  absent; BUG-E v1 descoped to the interim guard per D-11; facade+endpoints = named follow-up.**
- OQ-6: Does the review tree have mutating entry points beyond the context menu
  (keyboard shortcut, drag-rename)? (Phase-3 build enumerates; the guard must cover all —
  D-11.)

## Phase 0 — LIVE repro (Apollo, 2026-07-02 08:01–08:08 PT) — COMPLETE, D-9 gate satisfied

Driven over the real dashboard `/api/ws` (password-login → ws-ticket → JSON-RPC), exactly the
desktop's path, against the live gateway (pid 11197). Disposable seeded sessions (150/400 msgs),
deleted after. No code changes; no instrumentation added (nothing to revert).

**R1 — Worker-stale-history CONFIRMED.** Every `slash.exec /compress` returned
`"(._.) Not enough conversation to compress (need at least 4 messages)"` as `output` while the
seeded session held 150-400 messages — the slash worker's resumed copy is empty/stale. (§3 pt-2 ✓)

**R2 — Double-execution + lock contention CONFIRMED.** Two concurrent `/compress` (8s apart,
sid `20260702_080725_bb27d2`): second hit
`compression skipped: another path is compressing session=… (holder=pid=11197:tid=…) — returning
messages unchanged to avoid session fork` (agent.log 08:07:36.873) → warning "No changes …
(unchanged)". (§3 pt-4 hypothesis ✓)

**R3 — NEW MECHANISM (refines §3 pt-4): the no-op lock-skip POISONS the in-flight compression.**
The first compress genuinely compacted — `LCM compaction #1: 400 messages → 95` /
`compression done: 400->95` (agent.log 08:08:06) — yet its RPC response reported **"No changes
from compression: 400 messages."** Cause, from source: when `compress_context` returns the
input unchanged (lock-skip), `_compress_session_history` STILL swaps `session["history"]` and
**bumps `history_version`** (`server.py:2864-2869`). The in-flight first compress then fails its
own version check ("External mutation during compaction — drop the compressed result") and
**discards the real 400→95 result**. Exact match to Ace's live session `20260701_193329_d7608e`:
two presses (03:05:33, 03:05:53), one `⚠ Skipping concurrent` line, LCM compacted 355→194 at
03:06:45, UI reported "No changes: 355 / ~430,420 unchanged" — the result was dropped.
**Phase-1 scope addition (same function, no scope explosion): the no-change path must NOT swap
history / bump `history_version`** (`if removed == 0 / compressed is before_messages: return`
without write). Without this, even single-execution routing remains poisonable by a user retry
during a slow compress.

**R4 — BUG-A(b) CLASSIFICATION: ROUTING-CAUSED (D-6 GOOD case).** The engine works (LCM
compacted 355→194 and 400→95 when allowed to land). No engine fork needed; **AC-1b is owned by
this PRD.** Resolved `compression.enabled=true` (config ground-truthed; not a config issue).

**R5 — The 4018:** not reproduced over the raw WS (both invocations returned results, no
throw), consistent with the fallback being CLIENT-side: desktop `slash.ts` catches a
`slash.exec` throw (timeout — live compress took 38-74s — or worker-error) → retries via
`command.dispatch` → `_err 4018`. Phase 1 removes both the worker leg and the desktop fallback
for `compress`, killing the 4018 class regardless of which throw fires; the Phase-1 unit test
asserts the dispatch case directly.

**R6 — cosmetic (noted, non-gating):** the mirror's before/after token estimate inflates
"after" by exactly +16,038 (system prompt + tool schemas enter `_cached_system_prompt` between
the two estimates on a fresh agent) — e.g. "~282,409 → ~298,447" on a no-op. Fold into Phase 1's
one-coherent-report work.

**R7 — SLASH-COMMAND SWEEP (Ace: "common slash commands feel weird" — confirmed, same root).**
21 common commands driven over the live `/api/ws` against a seeded 20-msg session
(sid `2ccb4905` / `20260702_110508_3837a2`, deleted after). Verbatim results:

| Command | Verdict | Symptom |
|---|---|---|
| `/usage` | **STALE-WORKER** | "(._.) No active agent — send a message first." on a live session |
| `/history` | **STALE-WORKER** | "(._.) No messages in the current chat yet" with 20 messages present |
| `/prompt` | **STALE-WORKER** | "(._.) Empty prompt — nothing sent." (worker sees no live prompt state) |
| `/status` | partial-stale | Works, but reports "Tokens: 0" (worker's counter, not the live session's) |
| `/model` | UX bug | Returns **empty output** (no current-model echo) |
| `/clear` | UX bug | Dumps the full CLI splash banner (ANSI art) into the desktop chat |
| `/models`, `/rename`, `/effort` | not-a-command | "Unknown command" — disposition in the audit (alias or better error) |
| `/help` `/context` `/tools` `/skills` `/personality` `/title` `/save` `/queue` `/goal` `/snapshot list` `/memory` `/mem` | OK | Correct (incl. correct usage-error for bare `/queue`) |

**Class confirmation:** the "weird" feel = the slash worker's stale/empty resumed history
(R1's mechanism) leaking into every session-READING command — one root, not N separate bugs.
This **concretizes Phase 1's command-class audit (RC-8/INV-1):** the disposition list is now
measured, not hypothetical — `/usage`, `/history`, `/prompt`, `/status`(tokens) must be served
from the LIVE session (read-only live-serving is LOWER risk than compress: no mutation, same
seam); `/model` empty-echo and `/clear` banner-dump are surface UX fixes; the unknown-command
trio gets dispositioned in the audit table. No scope explosion — this was already Phase 1's
audit obligation, now with ground truth.

**D-9 disposition:** R1/R2/R4/R5 confirm §3 as written; R3 is an additive refinement that
CHANGES Phase-1's fix list (adds the no-bump-on-noop write guard). Per D-9 this requires a PRD
revision (this log + the Phase-1 addition below) + a fresh review pass for Phase 1 before build
— pass 9 dispatched for exactly that delta.

## Phase 0 — Static ground-truth (Apollo, 2026-07-02; STATIC portion, live repro still pending)

Cheap source/config probes done before any build (the live MBP↔Studio repro of BUG-A is the
remaining Phase-0 item):

- **OQ-1 (token):** CONFIRMED — Ace's `connection.json` has `{url, authMode:oauth}`, **no token**.
  → D-5 browser-URL path unavailable for his config; **bytes→temp (D-7) is the implementation.**
- **OQ-5 (`/api/fs/*` rename/trash):** CONFIRMED ABSENT — `web_server.py` exposes only
  `/api/fs/{list,read-text,write-text,read-data-url,git-root,default-cwd}`. **No rename, no
  delete.** → RC-13/RR-6 realized: BUG-E's "route through the facade" fix requires **building new
  destructive gateway endpoints** (auth + path-allowlist + repo-root binding + reversible-trash +
  its own review). **Decision (fold): BUG-E v1 = the RC-15 interim guard only** (disable remote
  Rename/Delete + D-4 message); the facade rename/delete + new endpoints become a **named
  follow-up PR**, not this PRD's Phase 3. Phase 3 keeps BUG-D (a/v ordering) + the D-8 lint rule +
  Reveal→honest-message. This SHRINKS the build and removes the destructive-endpoint blast radius.
- **RC-16 (cookie SameSite):** CONFIRMED SAFE — session cookie is `samesite="lax"`
  (`dashboard_auth/cookies.py`) + CORS locked to `localhost/127.0.0.1`. Cross-origin cookie-bearing
  side-effects aren't reachable → D-12's CSP/JS-disable is **belt-and-suspenders, not mandatory**
  (still cheap; keep it). The opaque-origin blob (D-5) remains required.
- **OQ-2 (compress "No changes") — STRONG static hypothesis, needs live confirm:** the slash
  worker subprocess (`slash_worker.py`, `HermesCLI(resume=session_key)`) and the live gateway agent
  share the **same `session_id`**, and the compression lock is keyed by `session_id`
  (`conversation_compression.py:485` `_lock_sid = agent.session_id`; acquire `:512`,
  skip-and-return-unchanged `:530-556` — same sub-ranges §3 cites). Both fire for a desktop
  `/compress` (worker via
  `process_command` + live
  via `_mirror_slash_side_effects`), so one grabs the lock and the other hits
  "another path is already compressing → returns messages unchanged" = the observed "No changes".
  → **BUG-A(b) is very likely routing-caused (D-6 GOOD case): fixing the double-execution should
  fix the no-reduction too.** Live repro on the MBP↔Studio pair is the honest final gate before
  claiming AC-1b here vs forking.
