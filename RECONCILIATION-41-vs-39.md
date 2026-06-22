# Reconciliation: 41 open PRs vs 39 combined-tree merges

No PRs are lost between the two counts. The arithmetic:

- **41** open arminanton PRs total.
- **− 1** = #50111 (the manifest/docs-only PR; excluded from the code combine by design).
- **= 40** candidate code PRs for the combine.
- The ordered 3-way combine **merged 39**; the 40th (#50296) was the lone overlap
  the no-resolve combine script aborted on. #50296 applies CLEAN individually on
  v0.17.0 (overlap is on agent_init.py, touched by ~6 prior PRs), and has now been
  rebased onto current origin/main (the agent_init.py drift conflict resolved by
  keeping both `_end_session_on_close` and `_persist_disabled`); #50296 is now
  GitHub-MERGEABLE (head 084b79bed).

So: 41 = 40 candidates + 1 manifest; 40 = 39 merged + 1 (#50296, now mergeable).
