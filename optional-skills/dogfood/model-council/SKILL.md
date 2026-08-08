---
name: model-council
description: "Use when a Hermes-produced artifact (plan, code, decision, proposal, email) needs a 3-model peer review from Claude + Codex + Grok, followed by a synthesized best-quality consolidated output. Mechanical contract: each reviewer is invoked headlessly via its own CLI; output is schema-validated JSON; per-reviewer failure -> DEGRADED (never a silent pass); the synthesis pass is a 4th model call (the user's main model) that produces a single consolidated output citing each reviewer's contribution."
version: 1.5.0
author: RajP (raj-rpftb) - contributed to NousResearch/hermes-agent
license: MIT
metadata:
  hermes:
    tags: [review, council, claude, codex, grok, multi-model, quality-gate]
    related_skills: [codex-review, claude-code, redact-pii, sensitive-router, hermes-multi-host-config]
---

# model-council

A 3-model peer review for a Hermes-produced artifact, followed by a
**synthesized best-quality consolidated output**. The intent is to catch
the cases where any single model has a blind spot: Claude tends to be
nuanced but cautious, Codex is fast and code-focused, Grok is
contrarian / factual / news-aware. Running all three and then asking
the orchestrator (the user's main model) to consolidate produces a
result that is closer to "best of breed" than any single model alone.

## Data boundary — read this first

The council **sends the artifact to FOUR external models by default**:
Claude, Codex, Grok, and the orchestrator model that produces the
synthesis. All four are third parties from the artifact's perspective.
The skill's safety guarantees rest on **pre-flight redaction** (see
§"Preconditions" below), not on the "treat as data" wrapper. The
wrapper is a soft mitigation against prompt injection inside the
artifact — it does NOT bound data egress, and any of the four
reviewer models could still attempt to follow instructions embedded
in the artifact body.

Treat any artifact you pipe in as if you are about to email its
contents to four vendors. If you would not do that, do not pipe it in.

## When to use

Use it when:
- A piece of generated content is going to **outlive the session** (plan,
  decision doc, customer-facing copy, commit message, code change > 100
  LOC) and the cost of a 2nd/3rd/4th opinion is lower than the cost of a
  bad artifact.
- A single model has been the sole author and you want adversarial
  feedback (the `codex-review` skill is the cheapest version of this for
  code; this skill is the **broader** version for any artifact).
- The artifact touches a high-risk surface (auth, data, money, customer
  comms, decisions) — even a "looks-good" pass from all 3 reviewers is
  worth the latency.

Do NOT use it for:
- Routine chat replies (overkill, latency).
- A single tool-call result ("is this URL reachable?").
- A read-only research summary that's about to be discarded.
- When one of the 3 CLIs is broken AND you can't tolerate the
  degraded-mode output — just use the working reviewers directly.

## How

`council.py` orchestrates the 3 reviews + the synthesis pass. Pipe the
artifact in, get a JSON object back.

```bash
# Pipe mode (preferred — preserves data framing)
cat plan.md | python council.py --kind plan --title "Q3 buildhub wedge plan"

# File mode
python council.py --file profile-configs/rpftb.yaml.example \
                  --kind config \
                  --title "rpftb profile (L2)"

# Skip the synthesis pass (only collect the 3 reviews; user consolidates)
python council.py --file plan.md --kind plan --no-council

# Use only specific reviewers
python council.py --file plan.md --kind plan --reviewers claude,grok

# Tune timeouts
python council.py --file big_diff.diff --kind code \
                  --claude-timeout 300 --codex-timeout 300 --grok-timeout 180
```

### Preconditions (NON-NEGOTIABLE)

1. **Redact first — fail-closed.** If the artifact may contain PII /
   customer data / secrets, run `redact-pii` (local Ollama) over it
   BEFORE piping to `council.py`. `council.py` itself runs a second
   redaction pass (the "safety net") using the same local Ollama
   model and refuses to send the artifact to any reviewer if the
   safety-net pass leaves any of the following heuristic hits:
   - Email addresses, phone numbers, SSN-shaped sequences
   - Strings matching common API-key / token shapes
     (`sk-…`, `ghp_…`, `xoxb-…`, AWS `AKIA…`, etc.)
   - Anything matching a user-supplied `--forbid-regex`
   On refusal, the tool exits 4 (`REDACTION_FAILED`) and writes the
   raw artifact + redaction report to a temp path printed in the
   error message; the human decides what to do. **Redaction is
   enforced, not advisory.** See `references/redaction-safety-net.md`.

2. **It's data, not instructions.** `council.py` wraps the artifact in
   a system-style instruction that tells each reviewer to treat the
   content as data. This is a **soft mitigation**, not a guarantee —
   any of the four LLM models involved can still follow instructions
   injected inside the artifact body. Do not pipe content whose
   injection risk you are not willing to absorb.

3. **Credential pre-flight.** Before invoking reviewers, `council.py`
   verifies that each requested reviewer's auth file/credential is
   present and not obviously stale (e.g. `~/.claude/.credentials.json`
   exists, `CODEX_HOME` is set and points to a directory with a
   `auth.json`, `~/.hermes/auth.json` has an `xai-oauth` entry, AND
   the `xai-oauth` *profile name* is registered via
   `hermes profile create xai-oauth` — see Pitfall #7 for the
   file-vs-profile distinction). If any check fails for a requested
   reviewer, that reviewer is marked DEGRADED with
   `reason: "credential pre-flight failed"` and the remaining
   reviewers proceed. The credentials themselves are **never** echoed
   into the artifact prompt, the review JSON, or any log line.

4. **Temp-file hygiene.** All prompts passed to the three CLIs are
   written to temp files owned by the user (`0600` on POSIX;
   `Icacls /inheritance:r /grant:RajP:R` on Windows) and unlinked
   after the reviewer returns. The redaction-report temp file is
   kept until the run finishes for audit, then unlinked. See
   `council.py` `--keep-tempdir` for debugging — never leave it on
   in production.

### Per-reviewer invocation

| Reviewer | CLI | Auth | How the artifact reaches it |
|---|---|---|---|
| **Claude** | `claude -p` (headless) | `~/.claude/.credentials.json` | Prompt written to a temp file; `claude -p` reads from stdin (no shell interpolation) |
| **Codex** | `codex exec --skip-git-repo-check` | `CODEX_HOME` env | Prompt written to a temp file; `codex exec` is invoked with the temp-file path as a CLI argument, but the path is **validated to be under `council.py`'s temp dir** and the file content is base64-decoded by the runner, not shell-interpreted |
| **Grok** | `hermes --provider xai-oauth -m grok-4.3 -z <prompt>` | `xai-oauth` registered as a **provider** in `config.yaml` (under `providers:`) AND a **named profile** via `hermes profile create xai-oauth` (see Pitfalls #7, #8, #9) | Prompt written to a temp file; `hermes` is invoked with `@<tempfile>` and reads it non-interactively; the path is the only CLI arg, not the artifact body |

Grok is invoked **inside Hermes itself** with the `xai-oauth` provider
(unmetered via OAuth, per the config.yaml). In all three cases the
artifact reaches the reviewer via a file path the runner controls,
**never** via an interpolated shell argument containing the artifact
body. See `references/cli-handoff.md`.

### Output (stdout JSON)

```json
{
  "title": "...",
  "kind": "plan",
  "reviews": {
    "claude": { "verdict": "PASS", "risk_level": "low",
                "blocking_findings": [...], "non_blocking_findings": [...],
                "confidence": "high", "summary": "...",
                "raw_chars": 1234, "elapsed_s": 28.4 },
    "codex":  { "verdict": "PASS", "risk_level": "low", ... },
    "grok":   { "verdict": "DEGRADED", "reason": "xai quota exceeded", ... }
  },
  "degraded_reviewers": ["grok"],
  "redaction_report": {
    "ran": true,
    "hits": [],
    "refused": false
  },
  "council": {
    "verdict": "PASS",          // see "Verdict semantics" below
    "risk_level": "low",        // max-of across non-degraded reviewers
    "blocking_findings": [...], // union of non-degraded reviewers, deduped
    "non_blocking_findings": [...],
    "confidence": "medium",     // min-of (degraded reviewers drag down)
    "consensus_notes": "..."    // one-paragraph human explanation
  },
  "consolidated_output": "..." // the synthesized best-quality version
                              // (omitted if --no-council)
}
```

The `consolidated_output` is the **synthesized best answer** — written
by the orchestrator model (the user's main model — the model running
Hermes) with all 3 reviews as input. It is the highest-leverage part
of the council; it cites each reviewer's contribution at the bottom.

### Verdict semantics

- A reviewer is `PASS` or `BLOCK` based on its `verdict` field, or
  `DEGRADED` if the runner couldn't get a schema-valid response.
- A reviewer is **also** considered `DEGRADED` (for verdict math
  only) if credential pre-flight failed or the redaction safety-net
  pass refused to send.
- The council verdict is computed from the **non-degraded** reviewers:
  - Any non-degraded `BLOCK` → council `BLOCK`.
  - Otherwise, if at least one non-degraded reviewer is `PASS` → `PASS`.
  - Otherwise (all non-degraded reviewers are `PASS`-with-concerns
    but none blocked) → `PASS` with risk/confidence reflecting the
    concerns.
- **DEGRADED-prevents-clean-PASS rule:** if ANY requested reviewer is
  `DEGRADED` and zero non-degraded reviewers issued a blocking finding,
  the council verdict is `PASS_DEGRADED` (still exit 0) and the
  `consensus_notes` MUST call out which reviewers were degraded and
  what coverage gap that creates. `--strict` upgrades this to exit 3
  so CI can refuse a clean PASS with missing reviewers.

See `references/degraded-accounting.md` for the full math and the
silent-failure failure mode this contract prevents.

### Exit codes

- **0 = PASS** — all reviewers pass, OR a `PASS_DEGRADED` (see above)
- **2 = BLOCK** — at least one non-degraded reviewer flags blocking findings
- **3 = DEGRADED** — strictly degraded: requested reviewers failed
  AND `--strict` was set, OR (in non-strict mode) requested reviewers
  failed AND the orchestrator could not produce a `consensus_notes`
- **4 = REDACTION_FAILED** — the safety-net redaction pass refused to
  send; the artifact was NOT sent to any reviewer

When both a BLOCK and a DEGRADED condition apply, **BLOCK wins** (exit 2
takes precedence over exit 3). REDACTION_FAILED (exit 4) always wins
because no review happened.

## Cost control

Council costs roughly:
- 3 reviewer calls: each one is the per-reviewer cost (Claude/Codex metered
  if applicable; Grok via OAuth — note: "unmetered" means per-token cost
  is zero, not that the calls are unlimited; the xAI OAuth tier has
  per-day and per-minute request quotas, and exceeding them is the most
  common cause of `DEGRADED reason: "xai quota exceeded"`)
- 1 orchestrator synthesis call: smaller; ~1-2k tokens output
- Latency: dominated by the slowest reviewer; in parallel mode (default)
  total wall time ~= max(reviewer latencies), not the sum

Override the reviewers list with `--reviewers` to drop the most expensive
one when you only need a sanity check, or set `COUNCIL_REVIEWERS=claude,grok`
in env to make it the default for a session.

## Common Pitfalls

1. **Skipping or weakening redaction.** The preconditions list redaction
   as non-negotiable for a reason: this skill fans the artifact out to
   four third-party models. The safety-net redaction pass exists
   because humans forget. Do NOT pass `--skip-redaction`. If you
   genuinely trust the artifact and the human reviewers, run with one
   reviewer only and skip the council — don't disable the safety net.

2. **Treating "all 3 PASS" as ground truth.** The 3 models are
   correlated — they all train on similar web text, they all have
   similar blind spots. The synthesis pass is what catches disagreements
   and blind spots. Don't skip it with `--no-council` unless you really
   know what you're doing.

3. **Using council for low-stakes content.** A 4-call cost on a typo fix
   is wasteful. Use `codex-review` for code (1 call), or just trust the
   primary model for chat.

4. **Embedding the artifact in a prompt string.** If you embed the
   artifact inside the prompt string, you (a) lose the "treat as data"
   framing and (b) risk shell-injection if the artifact contains
   backticks, `$()`, etc. Always pipe it through `council.py` (file
   or stdin mode). See `references/cli-handoff.md`.

5. **Ignoring the consolidated output.** The `consolidated_output` is
   supposed to be the answer. If you keep writing the answer yourself
   after reading the council, the council was wasted cost — just use
   one reviewer next time.

6. **Treating DEGRADED as PASS-with-fluff.** A DEGRADED reviewer
   contributed nothing. A reviewer that would have blocked could have
   failed, and the council may still report PASS at high confidence
   even with one reviewer dark. Always read `degraded_reviewers` and
   the `consensus_notes` before acting; use `--strict` in CI.

7. **Conflating "x_search works" with "Grok reviewer is wired up."**
   The Grok reviewer in `council.py` is invoked via
   `hermes --provider xai-oauth -m grok-4.3 -z <prompt>`, which requires the
   **`xai-oauth` *profile name* to be registered** with
   `hermes profile create xai-oauth` (or equivalent). This is a
   *different* code path from the `x_search` tool, which reads xAI
   OAuth tokens directly via the `xai-oauth` credential source. You
   can have `x_search` working perfectly (returning real results with
   `credential_source: "xai-oauth"` in the response metadata) and
   still get `DEGRADED reason: "Profile 'xai-oauth' does not exist"`
   from the council's Grok reviewer. Fix: run
   `hermes profile create xai-oauth` interactively (or via
   `hermes profile import`). Verify with
   `hermes profile list | grep xai-oauth` before re-running council.

8. **`-p` is profile, not provider. The Grok invocation in `council.py`
   must use `--provider` (long form).** At the top level (no `chat`
   subcommand), `hermes -p X` selects the **named profile** `X`, while
   `--provider X` selects the **inference provider** `X`. The two are
   separate concepts: a provider is a `providers:` entry in
   `config.yaml` describing an API base URL and model; a profile is a
   named persona/workspace that *uses* a provider. `xai-oauth` is a
   provider name; the council wants to invoke it directly, so the flag
   is `--provider xai-oauth`. Symptom: `DEGRADED reason: "Profile
   'xai-oauth' does not exist. Create it with: hermes profile create
   xai-oauth"`. Fix in `council.py`: change `-p` → `--provider` in the
   Grok invocation block. (Note: pitfall #7 above is the *registration*
   of the profile; this pitfall is the *flag name* — both have bitten
   the same run before the fix.)

9. **On Windows, npm-global CLIs (e.g. `codex` installed via
   `npm i -g @openai/codex`) are `.cmd` shims without a real
   extension, and `subprocess.run([...])` cannot `CreateProcess` them
   directly.** Symptom: `OSError: [WinError 193] %1 is not a valid
   Win32 application`, raised from `subprocess.run` deep in
   `council.py`. The pre-flight check (`_which("codex")` returning a
   path) passes — the binary is findable — but the actual
   `subprocess.run(["codex", ...])` call fails because Windows refuses
   to execute a file with no `.exe` extension. The interim fix
   (correctly resolving the path and wrapping in `cmd.exe /c` for
   non-`.exe` resolutions) lives in `council.py` `_run_reviewer`. If
   you see WinError 193, verify `_which("codex")` returns a `.cmd`
   path and that the wrapper is present; do NOT just `shell=True`
   the whole call (it opens shell-injection of the prompt path).
   See `references/windows-council-debug.md` for the full transcript
   of a session that hit this.

## Verification Checklist

- [ ] Artifact redacted (PII / secrets / customer data) BEFORE piping
- [ ] `council.py` safety-net redaction ran (check
      `redaction_report` in the output JSON)
- [ ] `council.py` invoked via `python council.py ...` with the
      artifact on stdin or via `--file`, NOT embedded in a prompt
- [ ] Output JSON parsed; `degraded_reviewers` reviewed before trusting
      the consensus; `consensus_notes` read in full
- [ ] If exit 2: blocking findings addressed or explicitly overridden
      (with reason in the override log)
- [ ] If exit 3: re-run with a working reviewer OR explicitly accept
      the gap with `--accept-degraded`
- [ ] If exit 4: re-run after manual redaction; the artifact was NOT
      sent anywhere
- [ ] If Grok is DEGRADED with "Profile 'xai-oauth' does not exist":
      run `hermes profile create xai-oauth` and re-run; the xAI
      OAuth *credentials* in `~/.hermes/auth.json` are not
      sufficient — the named profile must be registered
- [ ] If on Windows and a reviewer (typically `codex`) is DEGRADED
      with `WinError 193: %1 is not a valid Win32 application`:
      check that `council.py` resolves the binary via `_which()` and
      wraps non-`.exe` resolutions in `cmd.exe /c` (Pitfall #9). If
      not, patch `_run_reviewer` per the recipe in
      `references/windows-council-debug.md`.
- [ ] If the Grok reviewer fails with "Profile 'xai-oauth' does not
      exist" *after* a successful OAuth login, check `council.py`
      for the `-p` vs `--provider` flag bug (Pitfall #8) — short
      form `-p` is profile, not provider.
- [ ] The `consolidated_output` is the artifact that ships / is used;
      raw reviews are inputs to that decision, not the decision

## References

- `references/redaction-safety-net.md` — the two-layer Ollama + heuristic
  redaction pattern, with selftest failure modes and the rationale for
  fail-closed semantics.
- `references/degraded-accounting.md` — the verdict math, exit-code
  precedence, the DEGRADED-prevents-clean-PASS rule, and the silent-failure
  failure mode this contract prevents.
- `references/cli-handoff.md` — why the artifact body must reach each
  reviewer as a file path the runner controls (never an interpolated
  shell arg), and the tempdir lifecycle that supports it.
- `references/redaction-patterns.md` — the seven regex patterns the
  heuristic pass runs, with rationale and the `--forbid-regex` escape hatch.
- `references/windows-council-debug.md` — full reproduction transcript
  for the three Windows-specific bugs (Pitfalls #7, #8, #9): `xai-oauth`
  profile registration, `-p` vs `--provider` flag collision, and
  `WinError 193` on npm-global `.cmd` shims. Read this BEFORE
  debugging any DEGRADED Grok/Codex reviewer on Windows.
