# Async report rendering probe

Offline integration proof for Desktop async completion reports. The producer creates
synthetic output, passes it through the real cron manual-completion formatter,
process notification formatter, durable delivery and SQLite, then reads actual rows.
The browser imports production hydration, runtime, SystemMessage and Desktop CSS.
No model, scheduled job or user configuration is used.

From the repository root with the normal Python and npm dependencies installed:

```sh
export HOME=$(mktemp -d)
export HERMES_HOME="$HOME/.hermes"
export ASYNC_REPORT_ARTIFACT_DIR=$(mktemp -d)
# If Chromium is installed outside the temporary HOME, set
# PLAYWRIGHT_BROWSERS_PATH to that installation's browser cache.
.venv/bin/python evals/desktop_bug_campaign/async-report-producer.py "$ASYNC_REPORT_ARTIFACT_DIR/producer.json"
node evals/desktop_bug_campaign/async-report-vite.mjs
# In another shell with the same environment:
node evals/desktop_bug_campaign/async-report-live.mjs before
node evals/desktop_bug_campaign/async-report-live.mjs after
```

The Vite probe binds loopback port 18164. Stop it after verification. The pinned
baseline hydration is from commit `a688e7d5ff9aeaaa9c97d28c316467f89ab8c943`;
all other renderer modules are the current checkout. Compare the emitted JSON and
screenshots: cron/delegation/batch/legacy Markdown headings and tables are absent
before and present after, plain results survive, malformed envelopes remain compact,
and task goals/context/transcript footer paths do not render. Batch goals include a
multiline case because the producer does not flatten them.

This is producer/store integration plus live Chromium rendering, **not** scheduled
execution, completion-poller/HTTP/WebSocket E2E, Bot Chat routing, or native macOS QA.
The `process` case means an async-delegation event through the process notification
formatter, not a terminal-process completion event. All fixtures are synthetic.
