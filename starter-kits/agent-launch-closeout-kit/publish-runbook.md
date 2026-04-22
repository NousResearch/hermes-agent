# Publish Runbook — Agent Launch Closeout Kit

## Purpose
Freeze one canonical publish path for a proof-backed MVP so launch execution can happen without re-scoping the product.

## Week-one proof surface
Use `starter-kits/agentic-cron-orchestration-kit/` as the mandatory example case for this kit.

Honest claim to preserve:
- from a fresh notes context
- after injecting the exact note paths and workspace path into the prompt templates
- an operator can run preflight, schedule one recurring workflow, and execute the evening-doc-sync loop
- recorded clean-room proof time: **1.74 minutes**

Do not widen the claim to a fully proven four-job operating pack.

## Required source files
- `starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md`
- `starter-kits/agentic-cron-orchestration-kit/launch/demo-outline.md`
- `starter-kits/agentic-cron-orchestration-kit/launch/demo-capture-runbook.md`
- `starter-kits/agentic-cron-orchestration-kit/launch/demo-captions.srt`
- `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- `starter-kits/agentic-cron-orchestration-kit/launch/ship-note.md`
- `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`

## Canonical publish order
1. Run `bash starter-kits/agent-launch-closeout-kit/scripts/publish-preflight.sh` before touching browser publish or `x-cli` so missing auth or missing source files are surfaced explicitly.
2. In the same publish environment, open `https://x.com/home` or `https://x.com/compose/post` and verify the session is actually signed in. If X shows the logged-out landing page, a login redirect, or `Already have an account?`, treat the browser-session marker as stale, run `bash starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh --prepare`, and stop until the recovery packet has been completed with screenshot evidence.
3. Re-open the source launch thread and post it with no broader-claim edits.
4. Attach the strongest proof-backed asset available:
   - primary: short walkthrough cut using `demo-captions.srt`
   - fallback: screenshot or still from the proof artifact showing **1.74 minutes**
   - last fallback: screenshot of `Preflight OK`
5. Keep the CTA narrow: this is the fastest honest path to one recurring workflow, not a control plane.
6. Immediately record the post URL, timestamp, and attachment choice in `launch-execution-log.md`.
7. Sync the weekly pipeline note, CEO note, factory note, and ship checklist from the same execution log.

## Launch guardrails
- Never imply that unpublished demo capture blocks the product ship decision.
- Never widen the claim beyond the current proof artifact.
- Never hide the explicit path-injection requirement.
- If demo capture is still pending, publish against the proved claim anyway.
- Never treat `~/.hermes/state/x-access.json` as enough publish proof by itself; the live publish session still has to be signed in.
- Use `live-browser-auth-audit.md` whenever the marker and the real browser session disagree.
- Use `scripts/browser-auth-recovery.sh` as the canonical stale-marker recovery path and require a screenshot-backed proof artifact before publish resumes.

## Block-end verification
Closeout work does not count as "done" because the thread text or runbook exists. End the block with recorded proof in `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`.

### After the publish block
Before reporting that the launch thread is posted, run:

```bash
python3 - <<'PY'
from pathlib import Path
import re, sys
text = Path('starter-kits/agent-launch-closeout-kit/launch-execution-log.md').read_text()
missing = []
if re.search(r'- Status: pending publish\b', text):
    missing.append('publish status still pending')
for label in ['URL', 'Timestamp', 'Attachment used']:
    if re.search(rf'- {re.escape(label)}:\s*$', text, re.M):
        missing.append(f'{label.lower()} blank')
if missing:
    print('Publish block not closed:')
    for item in missing:
        print(f'- {item}')
    sys.exit(1)
print('Publish block closed with recorded proof.')
PY
```

### After the capture block
Before reporting that walkthrough capture is done, run:

```bash
python3 - <<'PY'
from pathlib import Path
import re, sys
text = Path('starter-kits/agent-launch-closeout-kit/launch-execution-log.md').read_text()
missing = []
if re.search(r'- Status: pending capture\b', text):
    missing.append('capture status still pending')
for label in ['Recording path', 'Duration', 'Edited asset path']:
    if re.search(rf'- {re.escape(label)}:\s*$', text, re.M):
        missing.append(f'{label.lower()} blank')
if missing:
    print('Capture block not closed:')
    for item in missing:
        print(f'- {item}')
    sys.exit(1)
print('Capture block closed with recorded proof.')
PY
```

If either command fails, the work is still status narration. Update the execution log first, then sync the weekly notes from that proof source.

## Live auth audit
- Latest audit: `starter-kits/agent-launch-closeout-kit/live-browser-auth-audit.md`
- Recovery packet script: `starter-kits/agent-launch-closeout-kit/scripts/browser-auth-recovery.sh`
- Current finding: `x-access.json` is currently `stale`, and the live Hermes browser session still reaches X's logged-out/login surfaces until a fresh signed-in proof is captured.
- Blocking condition to clear: signed-in composer or signed-in home surface in the actual publish session.

## Fill-before-publish fields
- Product name: Agentic Cron Orchestration Kit
- Launch thread source path: `starter-kits/agentic-cron-orchestration-kit/launch/launch-thread.md`
- Proof artifact path: `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- Primary attachment path:
- Fallback attachment path: `starter-kits/agentic-cron-orchestration-kit/qa/clean-room-proof-run-2026-04-17.md`
- CTA / destination:
- Publish URL after post:
- Publish timestamp after post:

## Completion record
After publish, update:
- `starter-kits/agent-launch-closeout-kit/launch-execution-log.md`
- `Projects/Hermes/Weekly MVP Factory.md`
- `Projects/Hermes/MVP Pipeline — Week of 2026-04-20.md`
- `Projects/Hermes/Agent Launch Closeout Kit — CEO Note.md`
- `Projects/Hermes/Agentic Cron Orchestration Kit — Ship Checklist.md`
with the exact URL, timestamp, and attachment used.
