# Ship Note — Agentic Cron Orchestration Kit

## Ship decision
Ship against the **starter-workflow claim**, not the full four-job-pack claim.

## Honest shipped claim
From a fresh notes context, an operator can:
- run the preflight successfully
- inject exact note/workspace paths into the prompt template
- schedule one recurring workflow
- execute the evening-doc-sync loop
- update durable notes/checklists from that workflow logic

Proof recorded:
- `qa/clean-room-proof-run-2026-04-17.md`
- elapsed proof time: **1.74 minutes**
- committed artifact: `dc4e3162`

## Why this is the right ship line
- It is the claim we actually proved.
- It still solves the painful problem: getting one autonomous recurring workflow moving without babysitting.
- It avoids overclaiming that the full four-job operating pack has been fully proven end-to-end when it has not.

## What is included in the shipped MVP
- README with explicit setup contract
- four workflow prompt templates
- four note/checklist templates
- local preflight script
- QA proof plan and starter-workflow proof artifact
- launch thread and demo outline tied to the verified claim

## What comes next after ship
- optional broader proof of the full four-job operating pack
- stronger examples for injecting note/workspace paths
- additional launch assets and operator examples
