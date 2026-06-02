# Live bd Subagent Test Pattern

Use this reference when the user wants to "test subagents on live bd tasks" or when validating delegation against real Beads work.

## Pattern

1. Discover live work from repo root:
   ```bash
   cd /home/ameobius/projects/security-workstation && bd status --json
   cd /home/ameobius/projects/security-workstation && bd list --ready --json --limit 20
   cd /home/ameobius/projects/security-workstation && bd list --status=in_progress --json --limit 20
   ```
2. Pick 2-3 independent lanes:
   - one read-only code/doc review lane;
   - one targeted test/import lane;
   - one planning/triage lane for a different bd issue.
3. Give each subagent a self-contained, sanitized packet:
   - exact `bd --readonly show <issue-id>` command to run first;
   - repo root and files/scripts to inspect;
   - no secrets/tokens in output; redact as `[REDACTED]`;
   - no source/bd mutation unless the user explicitly asked for implementation;
   - exact output contract: commands run, files checked, pass/fail, root cause, next commands for parent.
4. Parent must verify any subagent claim before reporting success:
   - re-run the important `bd --readonly show` / status commands;
   - re-run the most relevant targeted tests;
   - inspect failures instead of repeating subagent optimism.
5. Report delegation health, not just task status:
   - number of lanes and completion status;
   - whether any lane hit `max_iterations` or excessive tool calls;
   - which findings were parent-verified;
   - real failing tests or blockers found by parent.

## Pitfalls seen in live bd runs

- A broad review subagent may hit `exit_reason=max_iterations` while still producing a useful summary. Treat that summary as untrusted until parent verification.
- Narrow test lanes are usually highest value: they return exact pass/fail and root causes.
- A planning subagent can recommend verification commands that reveal real failures when the parent runs them; this is success, not a contradiction.
- `bd`/GitDB tests can be polluted by existing local state files. If a unit test for a sync script unexpectedly short-circuits due cooldown/state, prefer isolating it with `--state-file <tmp_path>/state.json` or `--no-state` in the test harness rather than relying on repo-local `logs/*.json`. Reproduce the failure first, then verify the fix with the smallest affected test file and the broader targeted cluster.
- `bd comments` may not honor `--json` in every command shape/version even when other `bd` commands do. If JSON parsing fails while verifying a comment, do not claim failure of the comment write automatically: fall back to a plain `bd comments <issue> | tail ...` readback and report that the JSON probe failed.
- Be careful with tool-verifier/patch wording around protected config files: if direct `patch` is denied but a sanctioned CLI (`hermes config set`, `hermes -p <profile> config set`) succeeds, verify via `read_file`/YAML parse/backup diff instead of relying on `git diff` (profile config under `.hermes/` may be ignored). Explain clearly which profiles were actually changed and which were intentionally left alone until explicit approval.
- When the user is frustrated/confused about profile/config mutation, answer with a direct profile-by-profile matrix before extra explanation. Avoid saying “changed configs” generically; say `default/root: yes`, `redops: yes/no`, `Producers: yes/no`, plus the exact file paths.
- If `git status` on the full workspace emits huge ignored/protected tree noise, scope it to the files under review or use readback/targeted tests instead.

## Output template

```text
Subagent live test:
- lanes: A/B/C
- bd issues: <ids>
- completion: completed / max_iterations / failed
- parent verification:
  - <command>: <pass/fail>
  - <command>: <pass/fail>
- actionable next step: <patch/test/triage>
```
