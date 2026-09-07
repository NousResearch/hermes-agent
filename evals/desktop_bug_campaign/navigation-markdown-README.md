# Navigation / Markdown campaign probe

This fixture mounts the production `MarkdownTextContent` component in Chromium and records its DOM, preprocessed text, screenshot and renderer errors. It is **component-level live renderer evidence**, not a full Desktop navigation or backend test. Global Desktop CSS is intentionally not loaded; hard versus soft break assertions inspect emitted `<br>` elements, not CSS wrapping. No backend or model request is made.

From the repository root, with the locked Desktop dependencies available:

```sh
node evals/desktop_bug_campaign/navigation-vite.mjs
DISPLAY=:208 PLAYWRIGHT_BROWSERS_PATH=/home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown/browsers node evals/desktop_bug_campaign/navigation-markdown-live.mjs before
```

The server uses port 18160. The campaign artifact directory is currently explicit in the runner; it is outside the source tree. Use a dedicated Xvfb display and isolated browser process. Vite's unique cache directory must include a `node_modules` segment: otherwise the React Compiler processes optimized dependency bundles and startup can stall.

Baseline source: `08b140d14e6c1d49f9b7ad02c9437fe940d54d65`. Observed two-space hard break loses both spaces and emits zero `<br>`; ordinary soft break also emits zero `<br>`; fenced code loses trailing spaces. This is a reproduction only, not a fix or a passing regression test.
