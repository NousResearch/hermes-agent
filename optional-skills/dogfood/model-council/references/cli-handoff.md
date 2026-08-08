# CLI handoff: artifact as file path, never shell-arg

council.py hands the artifact to each reviewer CLI via a file
path the runner controls. This reference documents the contract,
the rationale, and the temp-file lifecycle. The contract is
NON-NEGOTIABLE — it is the difference between "the artifact
reaches the reviewer safely" and "the artifact's contents can
sandbox-break, inject commands, or leak to unintended
destinations."

## The contract

For every reviewer invocation:

1. The artifact (plus reviewer prompt wrapper) is written to a
   per-run temp file, e.g. `tempdir/prompt_<reviewer>.txt`.
2. The reviewer CLI is invoked with the **temp-file path** as a
   CLI argument (or as a stdin source for `claude -p`).
3. The artifact body is **never** interpolated into a shell
   argument or environment variable.

Three concrete handoffs in council.py:

| Reviewer | CLI invocation | What reaches the reviewer |
|---|---|---|
| `claude` | `claude -p < tempdir/prompt_claude.txt` | prompt body on stdin (no shell arg) |
| `codex` | `codex exec --skip-git-repo-check tempdir/prompt_codex.txt` | prompt body inside the temp file, path is a validated file-system path |
| `grok`  | `hermes -p xai-oauth -m grok-4.3 -z @tempdir/prompt_grok.txt` | `@<path>` Hermes syntax reads the file non-interactively; the path is the only CLI arg |

All three are forms of "runner-controlled path → file content";
none of them is "runner-controlled arg → file content."

## Why this matters

If the artifact body reaches the shell as an interpolated
argument, three classes of bug open up:

1. **Shell injection.** A artifact containing backticks,
   `$(…)`, semicolons, or unbalanced quotes can execute
   arbitrary commands in the runner's shell. The "treat as
   data" framing is a soft mitigation for prompt injection; it
   does nothing for shell injection.
2. **Argument-length blowup.** Most CLIs have argv limits
   (Windows `CreateProcess` is the tightest at ~32k chars).
   Long artifacts passed via shell args will be silently
   truncated or rejected.
3. **Quoting hell.** A artifact containing literal
   double-quotes, single-quotes, and `$` (any typical
   shell script artifact!) will break shell-quoting in
   surprising ways, often silently. The first symptom is a
   reviewer that received an empty or partial prompt.

## Why `claude -p` uses stdin

`claude -p` reads its prompt from stdin. The runner opens the
temp file and feeds it to the subprocess's stdin, which is a
kernel-level file descriptor handoff. The artifact never
appears in `argv`, never in the process's command line, never
in any shell history. This is the cleanest of the three
handoffs.

## Tempdir lifecycle

```
mkdtemp(prefix="council_")        # at run start
   |
   v
atexit.register(shutil.rmtree)    # auto-clean on any exit
   |                              # (success, exception, signal)
   v
unless --keep-tempdir             # opt-in for debugging
```

Per-run tempdir permissions: `0600` (POSIX) where supported;
Windows uses the inherited user ACL. The redaction refusal path
preserves the raw artifact + redaction report in this tempdir
for human audit before atexit cleanup.

## The internal-inconsistency bug (and how it was fixed)

The v1.0 SKILL.md Grok row showed
`hermes -p xai-oauth -m grok-4.3 -z <prompt>` — the prompt
interpolated into a CLI argument. The v1.2 row shows
`hermes -p xai-oauth -m grok-4.3 -z @<tempfile>`, and the
prose explicitly states the path is the only CLI arg. The
table row and the prose now agree, and a future reader
checking one against the other will find them consistent.

## Verification recipe

```python
# Confirm the artifact body never appears in argv:
import subprocess, tempfile, pathlib
artifact = "rm -rf / # $(touch /tmp/owned) ; echo '`whoami`'"
tmp = pathlib.Path(tempfile.mkstemp(suffix=".txt")[1])
tmp.write_text(artifact)
# If you have a real CLI, run it with the path arg, not the body.
# Then check the subprocess command line via ps/Process Explorer:
# it should contain the TEMPFILE PATH, not the artifact body.
```

The skill loader (`s1max-hermes-ops`) already documents a
related Windows quirk: `subprocess.run(["codex", ...],
shell=False)` fails on Windows even when `codex` is on PATH,
because `CreateProcess` does not search PATH for bare names.
The fix is `shutil.which` to resolve the bare name to an
absolute path first (see `s1max-hermes-ops` →
`references/windows-cli-spawn.md`). council.py's `_which`
helper is a less-complete version of the same pattern —
sufficient for this skill's needs but worth replacing with
`shutil.which` if you ever copy council.py's spawn logic into
a more general orchestrator.
