# apps/desktop/src governance

Read repository-root `AGENTS.md` and its required routing table before acting.
These are supplemental current-lineage rules, not exceptions to global rules.
Read this file completely; automatic hints may truncate it. Scope follows the
behavior affected, including tests, docs and callers outside this directory.

### Electron Desktop Chat App (`apps/desktop/`)

A **separate** chat surface from both the classic CLI and the dashboard's embedded TUI. It is an Electron + React + nanostore renderer (`@assistant-ui/react`) that talks to a `tui_gateway` backend over JSON-RPC (`requestGateway(method, params)`). The WebSocket/JSON-RPC transport lives in the framework-agnostic `apps/shared` package (`@hermes/shared` — `JsonRpcGatewayClient` + WS URL helpers), which the web dashboard (`web/`) also consumes; **desktop has no build/runtime dependency on the dashboard frontend** — it spawns a headless `hermes serve` backend server (the same gateway `dashboard` serves, minus the browser UI entirely: `serve` sets `headless_backend=True`, so `cmd_dashboard` skips `_build_web_ui` AND exports `HERMES_SERVE_HEADLESS=1` so `mount_spa()` disables the SPA even if a stray `web_dist/` exists — only the JSON-RPC/WS/API surface is reachable). `dashboard` and `serve` share `cmd_dashboard`/`start_server` but are independent surfaces — neither launches the other. The one exception is a backward-compat *fallback*: `serve` is newer, so the desktop spawn (`electron/backend-command.ts` + `backendSupportsServe()` in `electron/main.ts`) detects whether the resolved runtime registers `serve` and, only when it does not (an older managed install / PATH `hermes` the app hasn't updated yet), rewrites the argv to the legacy `dashboard --no-open`. Without that, a new app against an un-upgraded runtime would crash on an unknown subcommand and brick every mid-upgrade user. It does NOT embed `hermes --tui` — it has its own composer, transcript, and slash-command pipeline. For scoped Desktop architecture, state, resolver, transport, and testing rules, read `apps/desktop/AGENTS.md`.

**Slash commands in the desktop app are curated client-side, then dispatched to the backend.** The pipeline:

- **Backend already provides everything.** `tui_gateway/server.py` `commands.catalog` (empty-query list) and `complete.slash` (typed-query completions) both include built-in commands, user `quick_commands`, AND skill-derived commands (`scan_skill_commands()` / `get_skill_commands()`). The desktop app does not need a new RPC to see skills.
- **The renderer curates via `apps/desktop/src/lib/desktop-slash-commands.ts`.** This is the load-bearing file. It holds `DESKTOP_COMMAND_SPECS` (the built-ins and their Desktop surfaces) plus `NO_DESKTOP_SURFACE` block-lists for terminal-only / messaging-only / picker-owned / settings-owned / advanced commands that should NOT clutter the desktop popover.
  - `isDesktopSlashCommand(name)` — gates **execution**. Returns true for built-ins AND for any non-built-in (skill / quick command), so typed extension commands run.
  - `isDesktopSlashSuggestion(name)` — gates **discovery/completion**. Used by BOTH completion paths in `app/chat/composer/hooks/use-slash-completions.ts` (empty-query catalog filter + typed-query `complete.slash` filter) and by `filterDesktopCommandsCatalog`.
  - `isDesktopSlashExtensionCommand(name)` — true when the command is NOT a known Hermes built-in (i.e. a skill or user quick command). Both suggestion and catalog-filter paths allow extensions through so skill commands surface in the palette. (Added when fixing "skill commands missing from the desktop slash palette" — the curated allow-list was silently dropping every skill/quick command from completions even though they executed fine when typed.)
- **Dispatch** lives in `app/session/hooks/use-prompt-actions/slash.ts` (`runSlash`): built-ins that the desktop owns (`/skin`, `/help`, `/new`, …) are handled locally or via `commands.catalog`; everything else goes to `slash.exec`, falling back to `command.dispatch` (which the gateway resolves into skill / alias / exec directives). A skill command resolves to `{type: "skill", message}` and is submitted as a normal prompt.

**Rule:** the desktop slash palette's curation is about hiding noise (terminal-only / messaging-only built-ins), NOT about hiding user-activated extensions. Skill commands and `quick_commands` are extensions the backend surfaces — they belong in completions. If you tighten `desktop-slash-commands.ts`, keep `isDesktopSlashExtensionCommand` flowing into both the suggestion and catalog-filter paths. Tests: from `apps/desktop`, run `npx vitest run src/lib/desktop-slash-commands.test.ts` (workspace dependencies are installed at the repo root).

---

### Bot Mode (`apps/desktop/src/plugins/hermes-bots/`)

The desktop "Bots" experience ships bundled in-tree. Each bot is a Hermes
agent **profile** with a persistent identity. Its design rests on one settled
invariant that has been regressed repeatedly, cost users real conversation
history each time, and is not open for re-litigation in a routine PR:

**One bot = ONE canonical forever-chat, identified by NAME.** The chat's one
and only identity is **(profile, session titled exactly "Bot Chat")** — the
state DB's UNIQUE(title) index makes that pair an exact registry of at most
one row. The full lifecycle when a bot row is clicked:

1. **Resolve the registry, every time.** Look up the profile's `Bot Chat`
   session by exact title via `session.list {title, include_hidden: true}`
   (indexed, window-free; hidden rows resolve because canonical chats are
   always hidden; compression lineages resolve to the live tip). Row exists →
   open it. That is the entire happy path.
2. **No row → create it,** titled `Bot Chat`, born hidden, kicked off with
   the bot's intro. Creation adopts-before-minting: it re-runs the registry
   lookup first, so a concurrent or pre-existing row is opened, never forked.
   (`set_session_title` silently drops conflicting titles — returns 0 rows —
   which is how the 2026-08 infinite fork loop started; adopt-before-mint is
   what kills it.)

**There is NO session-id pin.** The previous design stored a pointer in
`ui_meta['hermes-bots'].chat` and verified it per click; five hardening
waves (#88690, #90732, #90751, the #91791 revert, #92042) each guarded a new
way that pointer dangled or got stolen — rows[0] steals, `last_session`
adoptions, transient clears, drifted-title welds (a pin re-anchored onto a
cron session passed every guard). Name-as-identity removes the failure class:
a name cannot dangle, and a corrupted historical pointer simply never gets
read. Legacy `chat` keys in ui_meta are ignored and dropped from merges.

Why recency must never win (the #91791 → #92042 lesson): canonical Bot
Chats are **unconditionally hidden** from the Sessions sidebar, so the bot
row is the ONLY door to the forever-chat. A "newest visible session wins"
preference doesn't re-order two equivalent entry points — it walls the
entire relationship off behind a row that previews one session and opens
another, and any stray draft that catches a prompt captures the row.
Side-chats started via "New chat with this agent" are not plumbing-titled,
stay visible in the Sessions sidebar, and are reachable there; they are
never the bot row's target.

Corollaries for reviewers:

- There is no per-bot session browser, by explicit design (removed in
  #90732). Do not add one back.
- Reject any PR that reintroduces a stored session-id pointer as canonical
  identity — including "as a fallback tier" or "for verification". The
  registry lookup is the whole contract; pointers are how every prior
  incident started.
- Reject any PR that consults recency, visibility, or "where the user left
  off" for the bot row's target — reports that motivate such a change are
  almost always about side-chats, and the fix belongs in the Sessions
  sidebar (hide-sweep false positives), not in the bot row's target.
- The gateway reports the registry row per profile as `canonical_session`
  on `profiles.list` (resolved server-side by title); roster preview,
  activity signals, and the `/new`→`/compact` guard all read it, so preview
  identity and click identity are the same row by construction.

Regression tests encoding this contract:
`tests/canonical-chat-registry.test.mjs` (includes a tripwire asserting the
open path never reads or writes a stored pointer),
`tests/canonical-chat-creation.test.mjs`, `tests/hide-bot-chats.test.mjs`,
and `tests/tui_gateway/test_profiles_list_canonical_session.py`.

---

<!-- End of relocated baseline guidance. -->
