# Content-engine cron scripts

These files are the canonical sources for the content-engine `no_agent` cron
jobs. Hermes requires executable cron scripts to live below the active
`HERMES_HOME/scripts` directory, so deployment is an explicit copy step:

```bash
install -m 0755 scripts/content_engine/{x_morning_article,li_daily_package,x_quote_scout,x_thesis_incubator}.py ~/.hermes/scripts/
```

After copying, run each job's normal dry/manual invocation and then update or
resume the existing cron job. Repository changes do not deploy automatically.
The scripts import routing from `content_engine/llm_generate.py`; the active
Hermes profile's generated `model` and `fallback_providers` configuration is
therefore the only route chain used at runtime.
