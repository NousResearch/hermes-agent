# Redaction safety net for council.py

The council fans the artifact out to **four external models by
default** (Claude, Codex, Grok, and the orchestrator model). Redaction
is therefore the single most important contract in the skill — not
the JSON schema, not the synthesis prompt. This reference captures
the pattern, the failure modes found in selftest, and the
rationale for each design choice.

## The two-layer design

council.py runs redaction in two layers, in order:

1. **Ollama pass (semantic).** Local Ollama (`qwen3:30b-a3b` by
   default, URL
   `http://100.125.229.22:11434` via `OLLAMA_URL` env) rewrites the
   artifact with stable tokens (`<EMAIL_1>`, `<PHONE_1>`,
   `<SECRET_1>`, `<PERSON_1>`) replacing PII, secrets, customer
   data. Best-effort: if Ollama is down, the model is missing, or
   the call times out (90s), the pass returns `ran: false` and the
   runner falls back to the heuristic layer without blocking.
2. **Heuristic pass (syntactic).** A small set of regex patterns
   (`references/redaction-patterns.md` lists them) runs against the
   *post-Ollama* text. Any hit fails-closed: the artifact is NOT
   sent to any reviewer, the runner exits 4
   (`REDACTION_FAILED`), and the raw artifact + redaction report
   are preserved in the per-run temp dir for the human to inspect.

The user can pass `--skip-redaction` to disable layer 1, but layer
2 still runs (the heuristic is the fail-closed net — humans forget,
Ollama's rewrite isn't perfect, and the cost of a false-positive
refusal is much lower than the cost of an exfiltrated API key).

## Why this order matters (and why layer 2 catches what layer 1 misses)

In selftest on 2026-06-16, a test artifact with one email, one
phone, one GitHub PAT, and one AWS access key was rewritten by
Ollama to all `<..._1>` tokens in a single pass — heuristic hits
`[]`, exit 0. The same artifact with `--skip-redaction` was caught
by the heuristic on **all four** items: email regex, phone regex,
`ghp_…` regex, `AKIA…` regex. So in practice Ollama does most of
the work, but the heuristic is the safety net that catches:

- Secrets with shapes Ollama doesn't recognize (custom internal
  tokens, internal customer IDs, formats new enough that the
  model's training data didn't include them).
- Artifacts where Ollama was skipped or unavailable.
- The `--forbid-regex` user-supplied pattern: layer-2-only,
  applied last so it catches anything the first two layers let
  through.

## Why fail-closed, not warn-and-continue

The original SKILL.md v1.0 made redaction a human responsibility
("Redact first"). The council review's first BLOCK finding caught
that: redaction was *advisory*, not enforced, and the contract
encouraged shipping unredacted artifacts to 3+ third-party
models. The v1.2 contract makes redaction **enforced** in
council.py itself, with a non-zero exit code and a clear refusal
path. `--skip-redaction` exists for power users but is loudly
labeled `DANGEROUS` and logged in `redaction_report`.

## Tempdir hygiene on REDACTION_FAILED

When the heuristic trips:

- The raw (un-Ollama-rewritten) artifact is written to
  `tempdir/raw_artifact.txt` so the human can see exactly what
  would have been sent.
- The redaction report (with all heuristic hits) is written to
  `tempdir/redaction_report.txt`.
- The tempdir path is printed in stdout JSON under
  `note` and `redaction_report.artifact_kept_at`.
- The tempdir is unlinked on exit by default (atexit handler);
  pass `--keep-tempdir` for debugging.

This is the audit trail: every refusal is recoverable, but only
for the human who owns the tempdir. No remote logging.

## Things to NOT add to this pattern

- **Cloud-hosted redaction APIs.** The whole point of "local
  Ollama" is that the artifact never leaves the machine before
  redaction. Routing through a hosted PII-detector would defeat
  the purpose.
- **More regex patterns.** Each new pattern adds false-positive
  risk. The existing seven (email, phone, SSN, sk-, ghp_, xox*-,
  AKIA*) cover the vast majority of secrets that actually leak;
  the `--forbid-regex` escape hatch handles the long tail.
- **A "redact then send anyway" mode.** If a user truly can't
  redact, the right answer is "don't pipe it in," not "send
  redaction warnings and continue." This is the difference
  between council.py and a polite lint check.

## Verification recipe

```bash
# Layer 2 only, force the heuristic path
python council.py --file dirty.md --kind plan \
  --title "test" --reviewers claude \
  --skip-redaction --no-council --claude-timeout 3
echo "expect exit=4, stdout shows heuristic_hits populated"
```

If layer 2 doesn't trip on `dirty.md`, the regex set is wrong or
the test data is malformed (real test data needs valid key
shapes — `ghp_<20+ chars>`, `AKIA<16 chars>`, etc.). Don't paper
over a non-tripping heuristic by lowering its standards.
