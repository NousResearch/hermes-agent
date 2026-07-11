# Desktop App Subtree Instructions

This file scopes desktop guidance to `apps/desktop/` work. Root `AGENTS.md` still contains the non-negotiable project rules. Full reference: `docs/agent-context/tui-and-desktop.md`.

## Desktop architecture

- The Electron desktop app is a separate React/nanostore chat surface. It is not the dashboard and does not embed `hermes --tui`.
- Desktop talks to a headless Hermes backend over JSON-RPC via shared transport code in `apps/shared`.
- Keep desktop state local to desktop features unless it genuinely belongs in shared transport code.

## Slash commands

- Desktop slash-command curation lives in `apps/desktop/src/lib/desktop-slash-commands.ts`.
- Curation hides noisy built-ins; it must not hide user extensions. Skill commands and `quick_commands` must remain discoverable and executable.
- Preserve `isDesktopSlashExtensionCommand` through both suggestion and catalog-filter paths.

## Verification

- Run the repo-root Vitest coverage for `apps/desktop/src/lib/desktop-slash-commands.test.ts` when touching slash-command curation.
