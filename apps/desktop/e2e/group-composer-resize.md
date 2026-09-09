# Group composer native resizing

Both new-thread and reply-in-thread inputs use `GroupMentionInput`. Drag the
native bottom-right corner vertically. Width stays owned by the flex layout;
height is bounded by the existing 36px minimum and 40dvh maximum. Long drafts
scroll inside the textarea. Chromium owns the inline height, so controlled
value updates do not reset it. No new preferences, storage, event handlers,
authentication, backend, or profile capabilities are introduced. Height is
not persisted across component remounts or app restarts.

## Verification

From `apps/desktop`, after the repository-root workspace dependencies are installed:

    npm run typecheck
    npx vitest run --project ui src/plugins/hermes-bots/group-chat-parts.test.tsx src/plugins/hermes-bots/group-chat-view.test.ts src/plugins/hermes-bots/group-chat-timeline.test.tsx src/plugins/hermes-bots/group-panes.test.ts
    npm run build
    npx playwright test e2e/group-composer-resize.spec.ts --reporter=list
    CSC_IDENTITY_AUTO_DISCOVERY=false npm run builder -- --dir --publish never

The E2E uses an isolated Electron app, HOME, HERMES_HOME, userData, and mock
inference. It creates disposable bots and a room through the real UI. It
waits for creation toasts to stop covering the native resize corner, then
uses actual pointer input on both entry points. It checks growth/shrink,
minimum/maximum clamps, container-owned width, long-content scrolling,
height retention during typing, Enter submission, Shift+Enter newlines,
mention insertion, simulated IME key guards, and a smaller viewport.
The unit regression separately proves paste-event forwarding and height
retention; it is not evidence of an actual native drag or OS IME interaction.

With the original `resize-none` class restored, the E2E fails on the first
drag (36px unchanged). With the fix, both inputs grow from 36px to 136px.
Build/package verification is separate from installation or the user's
running app. This change does not authorize installation, publication, or
restarting that app. To roll back a local deployment, revert the focused
commit and rebuild; no stored data migration is needed.
