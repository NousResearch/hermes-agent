#!/usr/bin/env python3
"""Weekly self-eval reminder: prompts agents to log their self-evaluations."""

# P13: disabled-staging guard — exit early when cron is disabled
import os as _os, sys as _sys
if _os.environ.get("DRY_RUN") == "1":
    print(f"[DRY_RUN] {_os.path.basename(__file__)}")
    _sys.exit(0)
import datetime as dt

TZ = dt.timezone(dt.timedelta(hours=1))
now = dt.datetime.now(TZ)

print(f"🔄 Self-Eval Reminder · {now.strftime('%d/%m/%Y · %H:%M:%S')}")
print()
print("It's Friday. Time for this week's self-evaluation.")
print()
print("**Who should eval this week**")
print("• **Tier 2 (rolling):** Check the Logboard for this week's rotation")
print("• **Tier 1 (flagged):** Any agent tagged Refinement-Needed since last eval")
print()
print(f"Use the template at `{now.strftime('/home/kensei/.hermes/governance/self-eval-schema.md')}`")
print()
print("Save evals to `/home/kensei/.hermes/governance/logboard/`")
print("as `self-eval-{agent}-{YYYY-MM-DD}.md`")
print()
print(">> **Eval protocol**")
print("1. Agent writes self-eval using the schema template")
print("2. Saved to Logboard as a markdown file")
print("3. Denji-reviewer reads and categorises: keep / refine / escalate")
print("4. Refinement-Needed entries get Tier 1 review within the week")
print("5. Profile changes go on the Profile Change Ledger with a 2-week follow-up")