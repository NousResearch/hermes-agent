# Gemini Instructions — go-workflow

Follow `AGENTS.md`. This file exists so Gemini-style agents can discover the same repo-local workflow without Hermes-specific context.

Start with:

```bash
python3 scripts/next_task.py --validate
python3 scripts/next_task.py --list --limit 5
python3 scripts/next_task.py --claim --agent gemini
```

Show the max-5 task preview before claiming or doing work on every workflow-triggered run.

Then read the handoff, work only inside scope, verify, and finish with evidence.
