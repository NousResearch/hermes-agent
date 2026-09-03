# Team Hermes Desktop maintenance

This is the durable operating contract for Dad's customized Team Hermes Desktop.

## Canonical source and protected runtime

- Active protected repository: `D:\Bkash Agents\team-hermes-desktop-update-20260828`
- Active protected branch: `bkash/team-hermes-desktop-update-20260828`
- Historical checkout: `D:\Bkash Agents\team-hermes-desktop` on
  `bkash/team-hermes-desktop` contains separate uncommitted work. Do not reset,
  overwrite, or use it as an upstream-update target until that work is reviewed
  and reconciled.
- Updater-managed official installation: `C:\Users\<you>\AppData\Local\hermes\hermes-agent`
- Protected builds: `C:\Users\<you>\AppData\Local\hermes\desktop-builds\team-hermes-desktop-<timestamp>-<sha>`

Never implement custom UI changes directly in the updater-managed official installation. An official update may replace it. Make changes in the canonical repository and package a protected versioned build outside the official installation.

## Required workflow

1. Resolve the executable used by the running window and inspect `git status` before editing.
2. Preserve unrelated and dirty work. Use narrow patches and do not reset or overwrite user changes.
3. Add a focused regression test for every repaired behavior.
4. Run the focused tests, TypeScript check, applicable lint, `git diff --check`, production build, and Windows packaging.
5. Review the diff for credentials, generated artifacts, unrelated files, and unsafe scope before committing.
6. Commit the accepted checkpoint and create an annotated `team-hermes-desktop-checkpoint-*` tag.
7. Copy `win-unpacked` into a new protected versioned directory. Do not overwrite or delete the prior known-good build.
8. Write `build-manifest.json` beside the protected executable with the branch, commit, tag, build time, executable path and SHA-256, checks, and rollback build.
9. Repoint only Team Hermes shortcuts to the protected executable. Keep `Hermes Desktop - Official Update.lnk` connected to the official installation.
10. Relaunch and verify the exact process executable path and the changed behavior in the packaged app.

## Updating from upstream without losing the custom UI

Do not pull or reset the protected branch directly to upstream `main`. Use the
repository-owned, fail-closed update command:

```powershell
# Read-only preflight: fetches upstream unless -SkipFetch is supplied.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\team-hermes-desktop\update-protected.ps1

# Deliberate update: creates a rollback ref, merges upstream, and verifies the
# Team, Group Chats, message-card, flashcard, and protected-edition contracts.
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\team-hermes-desktop\update-protected.ps1 -Apply
```

The command refuses dirty or non-Team-Hermes branches. It merges instead of
resetting, so the custom commit history remains reachable. It stops on merge
conflicts or failed tests/typechecking/build, and it verifies that
`desktop-builds\current.json` was not changed. Therefore an official source or
backend update cannot silently replace the last known-good Team Hermes UI.

After a successful source update, package into a new versioned protected build,
write its manifest, and visually verify the changed contracts. Only then switch
`current.json`; retain the prior executable as rollback. Never point the custom
launcher at `apps\desktop\release` or the updater-managed official installation.

## Product contracts that must survive future changes

- Team Hermes has its own application identity and `%APPDATA%\Team Hermes Desktop` runtime state.
- Appearance is global across agent profiles; switching a profile must not change light/dark mode.
- User and agent message cards use the full responsive transcript lane and must not leave arbitrary percentage gutters.
- Group and direct-message cards retain clear sender identity, timestamps, solid theme-native surfaces, and hover feedback.
- Topic threads remain collapsible without replacing the normal transcript with duplicate topic-lens chrome.
- Team and Group Chats remain separate, ordered Team-first, and independently collapsible and expandable.
- Agent replies, inter-agent handoffs, and background-process output remain compact attributed flashcards with bounded disclosure.
- The composer grows to a bounded scrollable height and must not cover the transcript, footer, or title bar.
- HUD mode exposes an obvious drag grip and remains movable.
- Capacity-blocked group members wait and retry instead of silently disappearing.

## Authority boundary

Local inspection, implementation, tests, commits, tags, packaging, rollback preservation, shortcut switching, and live verification are allowed when Dad requests a Team Hermes change. Pushes, pull requests, merges, releases, deployment, public publishing, credentials, account changes, and destructive cleanup require Dad's fresh explicit approval.

## Handoff receipt

Return: status; failure mode; changed paths; exact checks and results; protected build path; manifest path; commit and tag; rollback handle; residual risk; and any exact approval still required. A source edit, passing unit test, or transport receipt alone is not live completion.
