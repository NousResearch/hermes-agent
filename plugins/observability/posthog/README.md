# PostHog Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it.

It emits PostHog AI Observability events:

- `$ai_generation` for each LLM/API call
- `$ai_span` for tool calls

PostHog creates trace groupings from the shared `$ai_trace_id` on those events.

## Enable

```bash
python -m pip install posthog
hermes plugins enable observability/posthog
```

Install the SDK into the same Python environment that runs Hermes. For a git
install this is usually the Hermes venv, so this also works from the Hermes repo
checkout:

```bash
./venv/bin/python -m ensurepip --upgrade  # only needed if pip is absent
./venv/bin/python -m pip install posthog
```

Or check the box in the interactive `hermes plugins` UI. `plugin.yaml` declares
`pip_dependencies: [posthog]`, and `hermes plugins enable observability/posthog`
will warn if that package is not importable from the current Hermes Python.

## Required credentials

Set these in `~/.hermes/.env`:

```bash
HERMES_POSTHOG_PROJECT_TOKEN=phc_...
HERMES_POSTHOG_HOST=https://us.i.posthog.com   # or https://eu.i.posthog.com / self-hosted
```

Without the SDK or project token the hooks no-op silently — the plugin fails
open. Placeholder-looking tokens log a one-time warning and are ignored.

## Verify

```bash
hermes plugins list                 # observability/posthog should show "enabled"
hermes chat -q "hello"              # then check PostHog AI Observability
```

Look for `$ai_generation` events under Activity, or in **AI Observability → Generations / Traces**.

## Optional tuning

```bash
HERMES_POSTHOG_DISTINCT_ID=hermes-agent       # fallback distinct_id
HERMES_POSTHOG_ENV=production                 # event property tag
HERMES_POSTHOG_RELEASE=v1.0.0                 # event property tag
HERMES_POSTHOG_SAMPLE_RATE=0.5                # sample 50% of events
HERMES_POSTHOG_MAX_CHARS=12000                # max chars per field
HERMES_POSTHOG_PRIVACY_MODE=true              # omit prompt/response content
HERMES_POSTHOG_SYNC_MODE=true                 # use SDK sync mode when available
HERMES_POSTHOG_DEBUG=true                     # verbose plugin logging
```

## Privacy

Set `HERMES_POSTHOG_PRIVACY_MODE=true` to avoid sending `$ai_input` and
`$ai_output_choices`. Tool arguments/results are still sent in compacted form;
use sampling or disable the plugin if those may contain sensitive data.

## Disable

```bash
hermes plugins disable observability/posthog
```
