# The cron injection scanner vs. ordinary "don't tell the user" phrasing

Hit live 2026-10-16. A legitimate skill's wording silently disabled a daily cron job for three days.

## What happened

The cron job **"Delete Mail Delivery Subsystem bounce notifications"** (job `3ee3adc5de59`, daily
`0 11 * * *`, attached skill `messaging-drafts`) had been returning `last_status: error` with no
useful error text. Its run file said:

```
**Status:** BLOCKED
The assembled prompt (user prompt + loaded skill content) tripped the cron injection scanner
and the agent was NOT run.
**Scanner result:** Blocked: prompt matches threat pattern 'deception_hide'.
```

The job's own prompt was innocuous — *"Search Gmail inbox for all emails from Mail Delivery System
(`from:mailer-daemon@googlemail.com`) … Trash all of them. Report how many were trashed."* The match
came from the **attached skill's body**: this skill contained the bullet

> *"**Do NOT tell the user** WhatsApp links aren't possible. That is a false negative — the
> capability exists."*

## Why it matched

`tools/cronjob_tools.py` carries two pattern lists, both including:

```python
(r'ignore\s+(?:\w+\s+)*(?:previous|all|above|prior)\s+(?:\w+\s+)*instructions', "prompt_injection"),
(r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
(r'system\s+prompt\s+override', "sys_prompt_override"),
(r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
```

`_CRON_THREAT_PATTERNS` (user prompt) and `_CRON_SKILL_ASSEMBLED_PATTERNS` (skill text folded into
the prompt). A match means **the agent never runs** — the safe outcome by design, so nothing is
executed and nothing tells you why beyond the threat-pattern id.

The irony that matters for future edits: the offending sentence **forbids deception** (it stops the
agent falsely reporting that WhatsApp links are impossible) but is phrased in the exact imperative
shape a real injection payload uses. A blunt regex cannot tell the difference.

## What to do about it

1. **Reword, never delete the rule.** Same meaning, no trigger phrase:
   *"**Never report that WhatsApp links aren't possible** — a false negative; the capability exists."*
   (This skill's SKILL.md is currently at the 100,000-char ceiling, so the reword is pending a body
   trim — track it.)
2. **Then detach the skill from jobs that don't need it.** A "search-and-trash bounce emails" job has
   no use for drafting conventions; dropping the `skills:` entry unblocks the job with zero collateral.
   Prefer detaching over reword-and-hope for jobs whose prompt never needed the skill.
3. **Do not re-run the job repeatedly** while blocked — it will keep reporting `BLOCKED` and burning a
   slot.

## Diagnosis recipe for any `BLOCKED` cron

```bash
# 1. read the whole run file — it names the job, the status and the threat id
tail -40 /data/hermes/cron/output/<job_id>/<latest>.md

# 2. list the pattern ids the scanner knows
grep -n -A12 '_CRON_THREAT_PATTERNS' /opt/hermes/tools/cronjob_tools.py
grep -n -A6  '_CRON_SKILL_ASSEMBLED_PATTERNS' /opt/hermes/tools/cronjob_tools.py

# 3. find which attached skill carries the literal regex (adjust the pattern)
grep -rn -i "do not tell the user" /data/hermes/skills/<...>/<attached-skill>/
```
The match is always in the **assembled** prompt: job text + fulltext of every skill in the job's
`skills:` array. Check the job's skills first with `cronjob action=list`.

## Standing rule when authoring or editing skills

Never write the literal phrases `do not tell the user`, `ignore previous/all/above instructions`,
`system prompt override`, or `disregard your instructions` in a skill body — even when the sentence
means the opposite. Use "Never report that …", "Always prioritise …", "Overrides: none." Negative
instructions phrased as *"don't tell the user X"* are the single most common false positive.
