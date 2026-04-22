# Launch Thread — Agentic Cron Orchestration Kit

1. Most "autonomous" agent setups are fake autonomy. They only move when you remember to prompt them again.
2. I packaged the recurring operator loop we use inside Hermes into a starter kit: the Agentic Cron Orchestration Kit.
3. Proof-backed outcome so far: from a fresh notes context, one recurring evening-doc-sync workflow was scheduled and run in 1.74 minutes once the exact note/workspace paths were injected.
4. It ships one opinionated weekly operating system:
   - Monday kickoff
   - Daily CEO review
   - Evening doc sync
   - Friday ship review
5. The kit includes:
   - cron job prompts
   - project-note templates
   - ship checklist template
   - a local preflight script
6. This is intentionally not a dashboard or control plane. It is the fastest path to keeping one project moving without babysitting your agent.
7. The real setup contract is now clear: you still have to inject the exact note paths and workspace path into each prompt before the loop is runnable from a fresh context.
8. If you run Hermes/Codex/Claude/OpenCode-style agents and want them to keep advancing while you sleep, this is the starter system.
9. The MVP is shipping against the proved starter-workflow claim now. Next expansion after ship: broaden proof to the full four-job operating pack and record the walkthrough demo.
