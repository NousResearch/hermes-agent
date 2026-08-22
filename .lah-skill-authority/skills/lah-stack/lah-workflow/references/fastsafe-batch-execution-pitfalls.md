# FastSafe Batch Execution Pitfalls

Observed 2026-08-05 on `CLOE_RETRIEVAL_ANSWER_SUFFICIENT_NATIVE_TOOLS_PRESERVATION_REPAIR_V1` (PR #701).

## Multi-file `grep -c` breaks batch parsing

`grep -c PATTERN file1 file2` prints one line PER FILE (`file1:0\nfile2:0`), NOT a
single count. A Python batch loop that does `int(out[-1])` on the output either
crashes (ValueError on `file2:0`) or — if wrapped in try/except — silently falls
back to `len(out)`, which is 2 for a clean two-file check → CHECK FAILED on every
multi-file check even though every count is 0.

Fixes:
- Run one file per `grep -c` call (`grep -c PATTERN file || true`), or
- Parse per-file lines: `for line in out: num = int(line.split(':')[-1])`, or
- Use `grep -cE 'PATTERN' file1 file2 2>/dev/null | cut -d: -f2 | paste -sd+ - | bc` for a total.

## Count ≠ violation: config constants and JS helper names

Raw grep counts on a source file produce false positives that a naive batch flags:

| Pattern | Legit match that must NOT fail the check |
|---------|------------------------------------------|
| `grep -c 'api.deepseek.com'` on chat-completions-service.js | `defaultBaseUrl: 'https://api.deepseek.com'` — provider CONFIG constant, not a live call. Tests inject `fetchImpl` mocks. |
| `grep -cE 'truncate'` (SQL migration scan) | `function truncate(text, max)` — the JS string-truncation helper, not SQL `TRUNCATE`. |
| `grep -c 'fetchImpl'` on a test file | injected mock boundaries (desired), not real LLM/provider calls. |

Rule: when a check flags, VIEW the matched lines (`grep -nE ... | head`) and classify
before declaring FAIL. A count of 1 on a config constant or a JS helper is a PASS
with context, not a violation. Only a real call site (fetch to a live URL, SQL
statement, deploy command) fails the check.

## Secret-scan false positives in test fixtures

A test file legitimately contains `DEEPSEEK_API_KEY: 'sk-test-key'` (mock env) and
`'***'` placeholder values. The secret scan regex `sk-[A-Za-z0-9]{20,}` does NOT
match `sk-test-key` (too short), but scan patterns that match any `api[_-]?key`
will. Keep secret scans scoped to: (1) the src diff, (2) real credential files —
test fixtures with obviously-fake keys are not secrets. Never echo a real key;
`***` placeholders in env setup are fine as long as no real fetch uses them.

## Operator scan-consent policy: composite pipelines get denied

Observed 2026-08-05 on `CLOE_EXISTING_RETRIEVAL_GOVERNOR_AND_BUSINESS_CAPABILITY_GRAPH_RUNTIME_WIRING_REPAIR_V1`. The operator's security policy requires the EXACT scan command to be shown BEFORE execution. A single terminal call bundling many greps (`grep -rn ... $FILES | head; grep ... ; ...`) was denied by the approval gate. Retrying the same composite is forbidden — do not rephrase-and-retry.

Fix pattern:
1. Present the exact scan command first (in the reply, before calling terminal): list each grep with its purpose (FastSafe check number).
2. Execute ONE simple single grep per call (explicit file list, `-nE`, no pipes), not a multi-command composite.
3. If no response/consent arrives, a single read-only simple grep is the safe middle ground; never run destructive cleanup (rm, git checkout -- bulk) without explicit consent — stage-ciblé (Gate 8) alone keeps untracked artifacts out of commits.
4. `EXIT=1` from `grep` means "no matches" = clean; echo the intent (`echo "SCAN_EXIT=$? (1 = clean)"`) so the result is unambiguous.

