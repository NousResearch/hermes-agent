# Demo Outline — Agentic Cron Orchestration Kit

## Demo promise
Show an AI operator going from a static project note set to one live recurring workflow that keeps state current without manual babysitting.

## Canonical demo path
1. Start with a project notes folder or Obsidian vault containing:
   - Weekly MVP Factory note
   - Current week pipeline note
   - CEO note
   - ship checklist
2. Run `bash scripts/preflight.sh`.
3. Create the starter jobs from `prompts/`, but first inject the exact note paths and workspace path for the project under test.
4. Manually execute **Evening Documentation Sync** against the project notes.
5. Show the notes updated with built/verified/blocker/next-move truth.
6. Show the next morning **Daily CEO Review** taking over from durable state instead of memory.

## Capture checklist
- terminal recording of preflight passing
- file diff or note update from evening-doc-sync
- timing from fresh context to first successful workflow run
- short voiceover or captions: "stop babysitting your agents"

## Ship gate
Do not present this demo as final until the clean-room timed run is recorded and the under-30-minute claim is supported by real elapsed time.
