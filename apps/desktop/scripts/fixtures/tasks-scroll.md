# Task-list clipping regression

From `apps/desktop`, after the locked workspace dependencies and Playwright Chromium are installed:

```sh
npm run test:task-scroll
```

The two tests mount the production `ComposerStatusStack` (including status grouping,
section collapse, task rows and the production stylesheet) with 14 and 20 synthetic
task records. Real Chromium wheel input must expose and hit-test the final row
inside the capped viewport, above a composer boundary. Collapse and re-expand must
preserve scrolling. No inline CSS treatment or `scrollIntoView` repairs the page.
The fixture uses no backend, model, native Electron bridge or user session data;
it is a renderer-component regression, not a native-app end-to-end test.

The runner serves only loopback port 18120 and cleans up its browser and Vite
server. Its Vite cache is private. Set `TASK_SCROLL_OUTPUT` to retain the JSON
geometry and screenshots in a chosen directory. Otherwise these are retained in
the printed test run's temporary directory. In shared campaigns wrap the command
with the campaign's `flock tests.lock`, and isolate `HOME` and `HERMES_HOME`.
