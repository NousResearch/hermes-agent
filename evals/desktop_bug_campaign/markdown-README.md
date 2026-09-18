# Markdown whitespace live probe

This fixture now runs Markdown ONLY: no producer.json, hydration, SystemMessage,
backend, model request, or notification delivery. The separate async-report lane
owns #101078. Historical combined producer harness: commit 07a4da338bc, files
navigation-markdown-{probe.tsx,live.mjs}; markdown-producer.py is unchanged.

From the worktree root (locked workspace dependencies installed):

```sh
node evals/desktop_bug_campaign/navigation-vite.mjs
PLAYWRIGHT_BROWSERS_PATH=/home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown/browsers node evals/desktop_bug_campaign/navigation-markdown-live.mjs after
```

Vite requires owned port 18160 (strictPort); verify the HTML references this
worktree's probe entry. Chromium is headless; no Xvfb/native Electron needed.
NAVIGATION_ARTIFACT_DIR overrides the default campaign receipt directory.

The harness asserts 30 cases: five actual ingress functions, each with hard/soft
breaks, first-line indented code, unfinished fence with terminal spaces, closed
code with a final-space line, and leading/trailing blank lines. It waits for real
Shiki, reads code textContent, clicks the code-card Copy control, and reads the
real browser clipboard. Code expectations include the Markdown parser's final LF;
we preserve its raw payload rather than arbitrarily trimming it. Nonzero exit
means a failed assertion or page error. JSON and screenshot are retained.

Unit invariants live in src/lib/markdown-whitespace.test.ts (exactly two tests).
Run all verification under campaign tests.lock. This is production-component
Chromium proof, NOT native Electron/backend/transport or macOS proof.
