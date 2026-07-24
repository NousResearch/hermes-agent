# Next-run prompt for a real `/improve deep` on hermes-agent

After plan 002's recon, the next advisor invocation that wants to do a **real, source-changing** audit on this repo should paste the following:

```
/improve deep hermes-agent --focus correctness,security,perf --no-direction --no-docs
```

Then, internally, the advisor will:

1. Read `.plans/002-full-repo-recon.md` first for boundaries.
2. Verify recon file matches reality (re-run `wc -l`, `git log --name-only`, etc.).
3. Compute fan-out per the 8-subagent table in plan 002.
4. Stop and ask which 1-3 findings to turn into plans (per Phase 3 default of 3-5).
5. Write plans to `.plans/003-...md` onward.

## Why this prompt is short

Most of the advisor's setup work was already done in plan 002. The remaining work is:
- Diff current state from the recon
- Run the audit
- Write plans

If the advisor sees something the recon missed, it should update plan 002's "Current state" section in a fresh-context subagent (per `closing-the-loop.md` Phase 4).

## Anti-patterns explicitly disallowed by plan 002

These were already rejected once; do not re-litigate them:
- `main.cjs` refactor
- `.venv/` bundled code modifications
- `optional-skills/research/*` template TODO removal
