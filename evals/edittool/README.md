# Edit-Tool Shape Audit

This deterministic trap battery compares Hermes' shipped `patch` matching
semantics with an anchored, unique-match `str_replace` control. It is a meter,
not a proposal to change the production tool ABI.

The tasks cover exact anchors, indentation drift, duplicate matches, two-hunk
edits, already-applied edits, and materially wrong anchors. The Hermes arm
calls `tools.fuzzy_match.fuzzy_find_and_replace` directly, so its result tracks
the matcher used by the production `patch` tool without changing tool code.

## Run

```bash
python3 evals/edittool/test_edittool.py
python3 evals/edittool/runner.py --label current-main
python3 evals/edittool/report.py evals/edittool/results/current-main.json
```

Treat an `applied` result as an edit that landed, `rejected` as a fail-loud
refusal, and `no_change` as an already-applied/no-op signal. Compare the two
arms by task rather than treating fuzzy acceptance as automatically good: the
purpose is to expose where it rescues harmless formatting drift and where the
anchored control deliberately refuses an edit.
