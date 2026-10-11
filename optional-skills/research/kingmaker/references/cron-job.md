# Cron job template

Fill <CRON_EXPR> and <DELIVER> from the user's answers, then create the job with `cronjob_manage` (action: create):

```
name:     kingmaker
schedule: <CRON_EXPR>
no_agent: true
script:   kingmaker-job.py
deliver:  <DELIVER>
```

- `no_agent: true` - no model runs so a check costs nothing; stdout is delivered
  verbatim, EMPTY stdout sends nothing so a healthy tick is silent, and a non-zero
  exit or timeout is raised as a broken job - which is why decided outcomes exit 0.
- `script` - must resolve inside `$HERMES_HOME/scripts/` (the scheduler enforces
  that at creation and again at run time); a .py script runs directly, no wrapper.
- `schedule` - must match the `cadence` recorded in answers.json.
- `deliver` - `origin`, `local`, or `platform:chat_id`.

Record the job id so a later reconfigure updates this job instead of creating a second one:

`python3 $HERMES_HOME/scripts/kingmaker-job.py --setup --cron-job <JOB_ID>`

Verify with `cronjob_manage` (action: run) - expect ok with empty output.

Exit codes:
```
0 ok, 2 page changed, 3 no eligible model or blocked identity,
4 smoke failed, 5 apply or verify failed, 6 concurrent manual change
```

Codes 2-6 print a notice and exit 0 so cron delivers the message; only exit 1
(an undecided failure, e.g. an un-fetchable page) rides the failure channel.
