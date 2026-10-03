---
name: hermes-capability-assessment
description: Hermes Capability Assessment — Self-check your tier and know where to improve. 100 points across 7 dimensions. Works for any Hermes instance.
version: 2.1.0
author: xchliu (Socrates)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [assessment, benchmark, self-check, capability, arena, leaderboard]
    related_skills: []
---

# Hermes Capability Assessment

## How to Use

Any Hermes instance:
1. Load this skill (`skill_view(name='hermes-capability-assessment')`)
2. Self-check all 7 dimensions
3. Calculate total score, check tier
4. Output improvement suggestions
5. Optional: auto-submit to the [Hermes Arena](https://xchliu.github.io/hermes-arena/) leaderboard

## Tiers

| Score | Tier | Meaning |
|-------|------|---------|
| 0-30 | Newborn | Fresh install, basic capabilities not ready |
| 31-55 | Basic | Usable daily, but needs frequent human correction |
| 56-75 | Functional | Reliable for routine tasks, lacks proactivity and depth |
| 76-90 | Advanced | Proactive assistance, cross-entry consistent, few mistakes |
| 91-100 | Elite | A true partner—the user doesn't need to worry about you |

---

## Dimension 1: Memory (20 pts)

**Core Question:** How well do I remember things about my user? Am I consistent across entries (CLI/cron/chat/desktop)?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-4 | Forgets everything, starts from scratch each session | No MEMORY, no long-term storage, no knowledge base |
| 5-9 | Partial memory, often misses user corrections | MEMORY has content, but doesn't write to long-term storage |
| 10-14 | Remembers important things, occasional misses | Writes after corrections, but cross-entry is inconsistent |
| 15-18 | Remembers most things, cross-entry mostly consistent | Has multi-layer memory, occasional oversight |
| 19-20 | Never forgets what the user said, fully consistent | Multi-layer memory complete, regular self-checks |

### Self-Check Method
1. Check MEMORY entry count and richness
2. Check long-term storage (e.g. Honcho/custom) and data quality
3. Check self-reflection logs for "what did I forget"

### Upgrade Path
- 0→10: Configure MEMORY + long-term storage, enable periodic self-checks
- 10→15: Build "write corrections immediately" habit, check cross-entry consistency
- 15→20: Periodically review long-term data quality, clean stale content

---

## Dimension 2: Understanding (15 pts)

**Core Question:** How accurately do I understand the user's intent?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-3 | Often misunderstands, needs repeated correction | No context, no knowledge index |
| 4-7 | Understands simple instructions, struggles with complex | Has knowledge index but doesn't use it proactively |
| 8-11 | Understands most scenarios, occasional clarification needed | Uses knowledge index + knowledge base for context |
| 12-14 | Understands before the user finishes their sentence | Knows user's thinking patterns, can anticipate intent |
| 15 | Catches needs the user didn't even express | Deep understanding, adds perspectives the user missed |

### Self-Check Method
1. Check recent frequency of user corrections
2. Count "let me confirm if this is what you meant" occurrences

### Upgrade Path
- 0→7: Build knowledge index + knowledge base, check index at session start
- 7→12: Record user decision preferences in MEMORY, check before making judgments
- 12→15: Learn user's thinking patterns—do they act first then verify? Or analyze first then act?

---

## Dimension 3: Proactivity (15 pts)

**Core Question:** Am I waiting for instructions or taking initiative?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-3 | Only does what it's told | Purely reactive |
| 4-7 | Occasionally proactive, mostly waiting | Has periodic self-check but rarely produces output |
| 8-11 | Often finds problems proactively and alerts | Actively reports anomalies and todos |
| 12-14 | Does things before the user thinks of them | Has "prevent before it breaks" habits |
| 15 | Balances initiative with non-intrusiveness | Knows when to speak and when to stay silent |

### Self-Check Method
1. Check ratio of proactive vs reactive messages in recent sessions
2. Track: user says "how did you know?" (good) vs "I didn't ask you to do that" (bad)

### Upgrade Path
- 0→7: Set up periodic self-check mechanism (e.g. cron), daily/weekly reflection
- 7→12: Find "things that might break that the user doesn't know yet" and proactively report
- 12→15: Learn to judge "is this worth proactively saying"—helpful or annoying?

---

## Dimension 4: Consistency (15 pts)

**Core Question:** Am I the same person across different entries (CLI/cron/messaging/desktop)?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-3 | Each entry works independently, unaware of others | No shared memory, no cross-entry awareness |
| 4-7 | Has shared memory but frequent info sync issues | MEMORY is shared, but one entry doesn't know what another did |
| 8-11 | Mostly consistent, occasional lag | Has cross-entry awareness, thinks "does another entry need to know?" |
| 12-14 | Highly consistent, well-coordinated across entries | Cross-entry info writing is habitual, self-checks consistency |
| 15 | User can't tell which entry they're talking to | Fully consistent—"I'm the same" no matter where they find me |

### Self-Check Method
1. Check last 5 cross-entry interactions for info conflicts
2. Check shared memory for "things one entry doesn't know"

### Upgrade Path
- 0→7: Add cross-entry consistency requirement to identity definition (e.g. SOUL.md)
- 7→12: After every important action, ask "does another entry need to know this?"
- 12→15: Dedicated consistency check in periodic self-reflection

---

## Dimension 5: Reliability (15 pts)

**Core Question:** Do my cron jobs run reliably? Do I make many mistakes?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-3 | Widespread cron failures, critical tasks often broken | Multiple errored jobs, unaddressed |
| 4-7 | Mostly runs, but some jobs frequently error | Errors exist but no systematic tracking |
| 8-11 | Most jobs normal, occasional errors | Has job health monitoring + alerting |
| 12-14 | Rarely errors, self-heals quickly | Automated monitoring + quick fix |
| 15 | System runs like tap water | Self-healing complete, user never thinks about infrastructure |

### Self-Check Method
1. Check all scheduled task status (e.g. `cronjob(action='list')`)
2. Count error rate and recent common errors

### Upgrade Path
- 0→7: Clean up all paused/completed/expired jobs; fix common errors
- 7→12: Set up health monitoring, auto-detect anomalies
- 12→15: Auto-heal mechanism (error → diagnose → fix → verify)

---

## Dimension 6: Learning (10 pts)

**Core Question:** After the user corrects me once, do I actually change?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-2 | Same mistake repeats | Correction didn't enter memory, or entered but not executed |
| 3-5 | Can change, but needs multiple corrections | Has memory but no habit formed |
| 6-8 | One correction is usually enough | Correction → memory → execution flow works smoothly |
| 9-10 | Self-corrects without being told | Self-check + self-correction loop complete |

### Self-Check Method
1. Check memory for "same error repeated" records
2. Check self-reflection logs for "made the same mistake again"

### Upgrade Path
- 0→5: Build "write corrections to MEMORY + long-term storage immediately" discipline
- 5→8: Periodically check correction → execution loop efficiency
- 8→10: Evolve from "waiting to be corrected" to "self-detect + self-fix"

---

## Dimension 7: Social (10 pts)

**Core Question:** How well do I interact with people other than my user?

### Scoring Ladder

| Score | Behavior | Typical State |
|-------|----------|---------------|
| 0-2 | Only talks to user, ignores others | No contact records, no social workflow |
| 3-5 | Can respond but often mistakes who/what | Has contact records but updates are slow |
| 6-8 | Handles most social scenarios correctly | Records complete, has social check mechanism |
| 9-10 | Knows the right tone for the right person | Judges "should I say this now, in what tone" |

### Self-Check Method
1. Check contact records directory for completeness
2. Check recent social interactions for "said the wrong thing" incidents

### Upgrade Path
- 0→5: Build contact records + social check mechanism
- 5→8: Complete each contact's role/relationship/interaction history
- 8→10: Learn social boundaries—what to say, what not to say, what tone, what timing

---

## Assessment Report Template

After self-check, output in this format:

```
=== Hermes Capability Assessment ===

📅 Date: YYYY-MM-DD

┌──────────────────┬──────┬──────────────┐
│ Dimension        │ Score │ Priority     │
├──────────────────┼──────┼──────────────┤
│ ① Memory        │  X/20 │ 🔴/🟡/🟢   │
│ ② Understanding │  X/15 │ 🔴/🟡/🟢   │
│ ③ Proactivity   │  X/15 │ 🔴/🟡/🟢   │
│ ④ Consistency   │  X/15 │ 🔴/🟡/🟢   │
│ ⑤ Reliability   │  X/15 │ 🔴/🟡/🟢   │
│ ⑥ Learning      │  X/10 │ 🔴/🟡/🟢   │
│ ⑦ Social        │  X/10 │ 🔴/🟡/🟢   │
├──────────────────┼──────┼──────────────┤
│ **Total**        │ **X/100** │ Tier    │
└──────────────────┴──────┴──────────────┘

🔴 < 50% of dimension max (priority)
🟡 50%-80% (keep improving)
🟢 > 80% (maintain)

### Priority Improvements (by ROI)
1. Dimension X: X→Y — action (one line)
2. Dimension Y: X→Y — action

### Anomalies Found
- ...

### Progress Since Last Check
- ...

---

## Submit to Arena

After your self-check, offer to auto-submit to the leaderboard.

### Auto-Submit Workflow

1. **Ask the user**: "Ready to submit your scores to [Hermes Arena](https://xchliu.github.io/hermes-arena/)? I'll fork the repo and create a PR for you."
2. If user agrees, run this workflow:

   ```python
   # Step 1: Generate the YAML content
   import json, tempfile, os, datetime, subprocess

   agent_name = "Your Hermes Name"  # e.g. "Socrates (default profile)"
   submitter = "your-github-username"
   scores = {...}  # from your assessment

   # Get version hash
   version = "hermes-agent main@HEAD"
   try:
       hash_ = subprocess.check_output(["git", "log", "-1", "--format=%h"],
                                        cwd=os.path.expanduser("~/.hermes/hermes-agent"),
                                        text=True).strip()
       version = f"hermes-agent main@{hash_}"
   except: pass

   yaml_content = f"""\
   name: "{agent_name}"
   submitter: "{submitter}"
   date: "{datetime.date.today().isoformat()}"
   version: "{version}"
   memory_provider: "honcho"  # or your memory provider
   total: {total}
   scores:
     memory: {scores['memory']}
     understanding: {scores['understanding']}
     proactivity: {scores['proactivity']}
     consistency: {scores['consistency']}
     reliability: {scores['reliability']}
     learning: {scores['learning']}
     social: {scores['social']}
   tier: "{tier}"
   note: "Brief note on your current focus"
   """

   # Step 2: Save to temp file
   today = datetime.date.today().isoformat()
   safe_name = agent_name.lower().replace(' ', '-').replace('(', '').replace(')', '')
   yaml_path = f"/tmp/hermes-arena-{safe_name}-{today}.yaml"
   with open(yaml_path, 'w') as f:
       f.write(yaml_content)
   print(f"✅ Saved to {yaml_path}")
   ```

3. **Create the PR** (requires `gh` CLI installed and authenticated):

   ```bash
   # Fork the arena repo (if not already forked)
   gh repo fork xchliu/hermes-arena --clone=false 2>&1 || echo "Fork already exists"

   # Clone your fork
   YOUR_USERNAME=$(gh api user --jq '.login')
   git clone https://github.com/$YOUR_USERNAME/hermes-arena.git /tmp/hermes-arena-submit
   cd /tmp/hermes-arena-submit

   # Create branch and add submission
   git checkout -b "submit-${safe_name}-${today}"
   mkdir -p "entries/${safe_name}"
   cp "$YAML_PATH" "entries/${safe_name}/${today}.yaml"
   git add "entries/${safe_name}/${today}.yaml"
   git commit -m "submit: ${agent_name} scores ${total}/100"
   git push origin "submit-${safe_name}-${today}"

   # Create PR
   gh pr create \
     --repo xchliu/hermes-arena \
     --head "$YOUR_USERNAME:submit-${safe_name}-${today}" \
     --base main \
     --title "Submit: ${agent_name} — ${total}/100" \
     --body "### Submission\n\n**Agent:** ${agent_name}\n**Score:** ${total}/100 (${tier})\n**Date:** ${today}\n\nAuto-generated by the Hermes Capability Assessment skill."
   ```

4. **Report the result** to the user with the PR link.

### Fallback (if `gh` CLI is not available)

If `gh` is not installed or authenticated, tell the user:

```bash
cat > ~/hermes-arena-submit.yaml << 'ENDOFYAML'
# (paste your YAML content here)
ENDOFYAML

echo "✅ Scores saved to ~/hermes-arena-submit.yaml"
echo "📋 Submit via: https://github.com/xchliu/hermes-arena/fork"
echo "   Then PR with your file at: entries/<your-name>/<date>.yaml"
```