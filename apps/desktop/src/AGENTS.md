# Desktop renderer contracts

The root guidance and `apps/desktop/AGENTS.md` apply.

## Backend boundary

Electron and React use the framework-neutral `apps/shared` JSON-RPC client to talk to `hermes serve`. Desktop owns its composer and transcript. It does not depend on the dashboard frontend or embed the TUI.

The app launches a pooled `hermes serve --port 0` per connection and profile. `serve` is headless and dies with the app. The detached messaging gateway survives. Preserve the narrow older-runtime fallback from `serve` to `dashboard --no-open` only when command discovery proves that `serve` is absent.

A process may host several profile homes. Every lifecycle, status, settings, and new-session route identifies its profile. Each tile stores its owner route. Test a scope fix both with the profile as launch home and as a secondary home in the same process.

## Slash commands

`hermes_cli/commands.py` owns built-in command metadata. `commands.catalog` and `complete.slash` already include built-ins, quick commands, and skill commands.

`apps/desktop/src/lib/desktop-slash-commands.ts` owns desktop curation. The generated `apps/desktop/src/lib/desktop-slash-registry.json` is the offline fallback and comes from `scripts/dump_desktop_slash_registry.py`.

- `isDesktopSlashCommand` gates execution.
- `isDesktopSlashSuggestion` gates both completion paths and catalog filtering.
- `isDesktopSlashExtensionCommand` identifies names unknown to the built-in registry.

Curation may hide terminal-only or messaging-only built-ins. It must not hide user quick commands or skills. Keep extensions enabled in both suggestion paths and dispatch them through `slash.exec`, then `command.dispatch`. A skill result becomes a normal prompt. Test changes with the owning Vitest file.

## Bot Mode

A bot is a profile. Its canonical forever-chat is the unique session in that profile titled exactly `Bot Chat`.

Resolve that registry row on every open with `session.list {title, include_hidden: true}`. If absent, create a hidden `Bot Chat`, but re-run the lookup before minting so concurrent creation adopts the existing row. Compression lineage resolves to the live tip.

The canonical identity is never a stored session-ID pointer, recency, visibility, or `last_session`. Legacy `ui_meta['hermes-bots'].chat` pointers are ignored and removed. Canonical Bot Chats stay hidden from the normal Sessions sidebar. Side chats remain visible and are never the bot row's target. `profiles.list.canonical_session`, roster previews, activity signals, and open behavior must all refer to the same title-resolved row.

Keep the contract tests under `apps/desktop/src/plugins/hermes-bots/` and `tests/tui_gateway/test_profiles_list_canonical_session.py` aligned.

## Free-tier renderer state

`$freeTierStatus` mirrors `free_tier.status` and refreshes with status and after sign-in. `deriveBillingView` handles `billing.free_tier` before `logged_in`. Every entry point opens the same single-owner sign-in dialog. Logged-out sign-in never substitutes a portal link. Dialog state maps directly to the poll route's structured status and reason, and user copy follows `apps/desktop/AGENTS.md`.

## References

- Desktop interaction rules: `apps/desktop/DESIGN.md`
- Shared backend protocol: `tui_gateway/AGENTS.md`
- Dashboard distinction: `web/AGENTS.md`
- Slash registry: `hermes_cli/AGENTS.md`