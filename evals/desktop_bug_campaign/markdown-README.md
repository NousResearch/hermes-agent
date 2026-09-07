# Markdown rendering probe

This uses the production Desktop components in a real Chromium renderer, without
starting an agent or making a model request. It is not a native Electron/backend
transport E2E test.

From the worktree root, with the campaign's locked dependencies and Xvfb ready:

```sh
.venv/bin/python evals/desktop_bug_campaign/markdown-producer.py /home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown/producer.json
node evals/desktop_bug_campaign/navigation-vite.mjs
# In another terminal:
DISPLAY=:208 PLAYWRIGHT_BROWSERS_PATH=/home/teknium/.hermes/cache/desktop-bugs-74848ed3/navigation-markdown/browsers node evals/desktop_bug_campaign/navigation-markdown-live.mjs after
```

The producer isolates both HOME and HERMES_HOME, saves a deterministic report via
`save_job_output`, formats it through the cron completion and process-notification
producers, persists an actual delivery row, and reads it back from SQLite. It does
not execute the scheduled job. The browser feeds that row through hydration,
runtime conversion, and `SystemMessage`, independently of the assistant Markdown
samples. JSON receipts and screenshots go to the campaign artifact directory.

The browser is a receipt collector: compare `hardBr`, `softBr`, `breaks`, `outputs`,
`errors`, and `producer.dom`. Hard-break samples should emit a `br`; the soft-break
control should not. The separate async report loss (#101078) is not repaired by
the whitespace fix (#97117).
