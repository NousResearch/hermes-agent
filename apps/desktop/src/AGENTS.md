# apps/desktop/src/ — backend contract, slash palette, Bot Mode

Applies on top of `apps/desktop/AGENTS.md` (the judgment guide) and the root `AGENTS.md`.
Root TypeScript style rules apply.

## The desktop is its own chat surface on a `hermes serve` backend

Electron + React + nanostores (`@assistant-ui/react`) talking to a `tui_gateway` backend over
JSON-RPC (`requestGateway(method, params)`); transport lives in the framework-agnostic `apps/shared`
(`@hermes/shared`: `JsonRpcGatewayClient` + WS URL helpers), which the web dashboard also consumes.
The desktop has **no build/runtime dependency on the dashboard frontend**: it spawns a headless
`hermes serve` (`headless_backend=True` → `cmd_dashboard` skips `_build_web_ui` and exports
`HERMES_SERVE_HEADLESS=1` so `mount_spa()` disables the SPA even if a stray `web_dist/` exists).
`dashboard` and `serve` share `cmd_dashboard`/`start_server` but neither launches the other. It does
NOT embed `hermes --tui` — own composer, transcript, slash pipeline.

**One backward-compat fallback:** `serve` is newer, so the spawn (`electron/backend-command.ts` +
`backendSupportsServe()` in `electron/main.ts`) checks whether the resolved runtime registers `serve`
and ONLY when it does not (older managed install / PATH `hermes` not yet updated) rewrites argv to
legacy `dashboard --no-open`. Without it a new app against an un-upgraded runtime crashes on an
unknown subcommand and bricks every mid-upgrade user. Keep it narrow and tested.

Lifecycle: `serve` dies with the app by design; the messaging gateway survives it (spawned detached
via `/api/gateway/*`). Never re-parent the gateway under the backend — `gateway/AGENTS.md`.

## Slash commands: curated client-side, dispatched to the backend

- The backend already provides everything: `commands.catalog` and `complete.slash` include built-ins,
  user `quick_commands`, AND skill-derived commands. No new RPC is needed to see skills.
- `src/lib/desktop-slash-commands.ts` is the load-bearing file: `DESKTOP_COMMAND_SPECS` (built-ins
  and their desktop surfaces) + `NO_DESKTOP_SURFACE` block-lists (terminal-only / messaging-only /
  picker-owned / settings-owned / advanced). `isDesktopSlashCommand(name)` gates **execution** (true
  for built-ins AND any non-built-in so typed skill/quick commands run);
  `isDesktopSlashSuggestion(name)` gates **discovery** — used by BOTH completion paths in
  `app/chat/composer/hooks/use-slash-completions.ts` and by `filterDesktopCommandsCatalog`;
  `isDesktopSlashExtensionCommand(name)` is true for anything not a known built-in, and both
  suggestion and catalog paths let extensions through (the allow-list once silently dropped every
  skill/quick command from completions even though they executed when typed).
- Dispatch: `app/session/hooks/use-prompt-actions/slash.ts` (`runSlash`) — desktop-owned built-ins
  (`/skin`, `/help`, `/new`, ...) locally or via `commands.catalog`; everything else `slash.exec` →
  `command.dispatch` fallback; a skill command resolves to `{type: "skill", message}` and is
  submitted as a normal prompt.

**Rule:** palette curation hides noise (terminal-only / messaging-only built-ins), NEVER
user-activated extensions. If you tighten `desktop-slash-commands.ts`, keep
`isDesktopSlashExtensionCommand` flowing into both paths. Test: from `apps/desktop`,
`npx vitest run src/lib/desktop-slash-commands.test.ts` (workspace deps install at the repo root).

## Bot Mode (`src/plugins/hermes-bots/`) — one bot = ONE canonical forever-chat, identified by NAME

Each bot is a Hermes **profile** with a persistent identity. This invariant regressed repeatedly,
cost users conversation history each time, and is not open for re-litigation in a routine PR.

The chat's only identity is **(profile, session titled exactly "Bot Chat")**; the state DB's
UNIQUE(title) index makes that pair a registry of at most one row. Clicking a bot row:
1. **Resolve the registry, every time:** `session.list {title, include_hidden: true}` (indexed,
   window-free; hidden rows resolve because canonical chats are always hidden; compression lineages
   resolve to the live tip). Row exists → open it. That is the whole happy path.
2. **No row → create it**, titled `Bot Chat`, born hidden, kicked off with the bot's intro. Creation
   adopts-before-minting: it re-runs the lookup first so a concurrent/pre-existing row is opened,
   never forked (`set_session_title` silently drops conflicting titles — returns 0 rows — which is how
   the 2026-08 infinite fork loop started).

**There is NO session-id pin.** The old design stored a pointer in `ui_meta['hermes-bots'].chat`;
five hardening waves (#88690, #90732, #90751, the #91791 revert, #92042) each guarded a new way it
dangled or was stolen (rows[0] steals, `last_session` adoptions, transient clears, a pin re-anchored
onto a cron session). A name cannot dangle; legacy `chat` keys in ui_meta are ignored and dropped.
**Recency must never win** (#91791 → #92042): canonical Bot Chats are unconditionally hidden from the
Sessions sidebar, so the bot row is the ONLY door — "newest visible session wins" walls the whole
relationship off behind a row that previews one session and opens another. Side-chats ("New chat
with this agent") are not plumbing-titled, stay visible in the sidebar, and are never the row's target.

Reviewer corollaries: no per-bot session browser (removed in #90732; don't add it back). Reject any
stored session-id pointer as canonical identity — including "as a fallback tier" or "for
verification". Reject anything consulting recency/visibility/"where the user left off" for the row's
target; such reports are about side-chats and the fix belongs in the Sessions sidebar hide-sweep.
The gateway reports the registry row as `canonical_session` on `profiles.list` (resolved server-side
by title); roster preview, activity signals, and the `/new`→`/compact` guard all read it, so preview
identity and click identity are the same row by construction. Contract tests:
in `src/plugins/hermes-bots/`: `canonical-chat-registry.test.ts` (tripwire: the open path never
reads/writes a stored pointer), `canonical-chat-creation.test.ts`, `canonical-chat-adopt-on-conflict.test.ts`,
`bot-row-opens-canonical-chat.test.ts`, `hide-bot-chats.test.ts`; plus repo-root
`tests/tui_gateway/test_profiles_list_canonical_session.py`.
