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
python3 evals/edittool/runner.py --label <verified-commit>
python3 evals/edittool/report.py evals/edittool/results/<verified-commit>.json
```

The scorecard preserves `passed` v1 as its **status** metric: `applied` means
the matcher reported an edit, `rejected` a fail-loud refusal, and `no_change`
an already-applied/no-op signal. `no_change` and the strict arm's `rejected`
are intentionally distinct no-change results; the latter is not a Hermes
loss. The v1 `artifact_correct` metric separately checks the resulting file
contents, and records partial writes so a final `rejected` cannot hide an
earlier hunk that changed a file.

Every scorecard records the checked-out measurement, fixture, and matcher
commit along with the local Python/platform environment. This is a
deterministic matcher audit with no model inference: it measures neither the
complete `patch` wrapper nor model-facing tool-schema performance. In
particular, it does not substitute for the separate small-model experiment.

Compare the two arms by task rather than treating fuzzy acceptance as
automatically good: the purpose is to expose where it rescues harmless
formatting drift and where the anchored control deliberately refuses an edit.
