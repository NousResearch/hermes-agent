# Debugging model-council on Windows

This is a **session-specific reproduction recipe** for the three Windows bugs
that hit model-council's Grok and Codex reviewers. Read this BEFORE
spending time on `DEGRADED` debugging on Windows — the symptoms are
specific and the fixes are mechanical.

## Symptom 1: `Profile 'xai-oauth' does not exist` (Grok DEGRADED)

**Error message (verbatim from council.py stderr):**
```
grok rc=1: Error: Profile 'xai-oauth' does not exist. Create it with: hermes profile create xai-oauth
```

**Why it's confusing:** the `x_search` tool works perfectly in the same
session. Response metadata shows `credential_source: "xai-oauth"`. So
the OAuth credentials are clearly valid. What's missing is the *profile*.

**The two distinct concepts:**

| Concept | What it is | Where it lives | Verified by |
|---------|-----------|----------------|-------------|
| **Provider** | An inference API definition (base_url, model) | `providers:` in `config.yaml` | `hermes config show` |
| **Profile** | A named persona/workspace that uses a provider | `~/.hermes/profiles/<name>/` | `hermes profile list` |
| **Credential** | OAuth tokens / API keys for that provider | `~/.hermes/auth.json` | direct file read |

Council's Grok invocation goes through the *provider* path, not the
credential-source path that `x_search` uses. So:

- ✅ `x_search` working → credentials are valid
- ❌ Profile not registered → Grok reviewer DEGRADED

**Fix sequence (do all three):**

```bash
# 1. Register the xai-oauth provider in config.yaml
hermes config set providers.xai-oauth.base_url https://api.x.ai/v1
hermes config set providers.xai-oauth.model grok-4.3

# 2. Create the named profile
hermes profile create xai-oauth

# 3. Verify
hermes profile list | grep xai-oauth
hermes --provider xai-oauth -m grok-4.3 -z "Reply: PONG"
```

## Symptom 2: `-p` vs `--provider` flag collision

**Error message:** identical to Symptom 1, even after creating the
profile.

**Why it happens:** at the top level (no `chat` subcommand), `hermes`
treats the short flag `-p` as **profile name**, not provider. The long
form `--provider` selects the inference provider. Council.py's Grok
invocation originally used `-p` (the SKILL.md table is now updated to
use `--provider`):

```python
# WRONG — interpreted as profile name
["hermes", "-p", "xai-oauth", "-m", "grok-4.3", "-z", f"@{prompt_path}"],

# RIGHT — inference provider
["hermes", "--provider", "xai-oauth", "-m", "grok-4.3", "-z", f"@{prompt_path}"],
```

**Verify which flag you need:** `hermes chat --help` shows
`--provider PROVIDER`. The top-level `hermes --help` lists `-z PROMPT`,
`-m MODEL`, and a positional `{chat, ...}` subcommand tree. `-p` is
NOT listed at the top level — it's silently inherited from a different
parser and means "profile."

**Distinguishing Symptom 1 from Symptom 2:** if `hermes profile list |
grep xai-oauth` shows the profile but Grok still DEGRADES, you're on
Symptom 2. Patch `council.py` `_run_reviewer` to use `--provider`.

## Symptom 3: `WinError 193: %1 is not a valid Win32 application` (Codex DEGRADED)

**Error message (full traceback tail):**
```
File "...\council.py", line 273, in _run_reviewer
    proc = subprocess.run(
  File "...\subprocess.py", line 548, in run
    with Popen(*popenargs, **kwargs) as process:
  File "...\subprocess.py", line 1026, in __init__
    self._execute_child(args, executable, preexec_fn, close_fds,
  File "...\subprocess.py", line 1538, in _execute_child
    hp, ht, pid, tid = _winapi.CreateProcess(executable, args,
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
OSError: [WinError 193] %1 is not a valid Win32 application
```

**Why it happens:** on Windows, npm-global CLIs (e.g. `codex` from
`npm i -g @openai/codex`) install as `.cmd` shims **without** a
`.exe` extension. `subprocess.run(["codex", ...])` calls
`CreateProcess` directly, which refuses to execute files without
`.exe`. The pre-flight `_which("codex")` returns a valid path
(typically `C:\Users\<user>\AppData\Roaming\npm\codex`) but the actual
exec fails.

**Why `shell=True` is the wrong fix:** it opens shell-injection of the
prompt path. If the artifact (or `prompt_path` constructed from it)
contains `$()`, backticks, or `&`, you get command execution as a
side effect.

**The correct fix in `council.py`:** resolve the binary via `_which()`,
and wrap in `cmd.exe /c` only when the resolved path doesn't end in
`.exe`:

```python
def _win_exec(bin_name: str, fallback: str) -> list[str]:
    """Resolve a CLI; on Windows, wrap non-.exe (npm-global .cmd) in cmd.exe /c."""
    resolved = _which(bin_name) or fallback
    if os.name == "nt" and not resolved.lower().endswith(".exe"):
        return ["cmd.exe", "/c", resolved]
    return [resolved]

# In _run_reviewer:
if reviewer == "claude":
    proc = subprocess.run(
        _win_exec("claude", "claude") + ["-p"],
        stdin=prompt_path.open("r", encoding="utf-8"),
        capture_output=True, text=True, timeout=timeout,
    )
elif reviewer == "codex":
    proc = subprocess.run(
        _win_exec("codex", "codex") + ["exec", "--skip-git-repo-check", str(prompt_path)],
        capture_output=True, text=True, timeout=timeout,
    )
elif reviewer == "grok":
    proc = subprocess.run(
        _win_exec("hermes", "hermes") + ["--provider", "xai-oauth", "-m", "grok-4.3", "-z", f"@{prompt_path}"],
        capture_output=True, text=True, timeout=timeout,
    )
```

**Why this is safe:** `cmd.exe /c <resolved_path> <args...>` doesn't
re-parse the path through the shell. The args list is still
`subprocess.run`'s argv (not a single command line), so prompt_path is
not subject to shell metacharacter expansion. Windows just uses
`cmd.exe` as the trampoline that knows how to execute `.cmd` files.

## Full reproduction transcript (excerpted)

The actual fix sequence on 2026-06-20, after the user asked
"do 1 and 2" (referring to "fix codex and the xai-oauth profile"):

```text
# Pre-flight reports all green even when reviewers fail:
{"claude": {"ok": true, "on_path": true},
 "codex":  {"ok": true, "on_path": true},
 "grok":   {"ok": true, "on_path": true}}

# First failure: Grok DEGRADED with "Profile 'xai-oauth' does not exist"
# Codex DEGRADED with "codex CLI not found: [WinError 2]"

# Diagnosis 1: `x_search` works in the chat (so creds are valid)
# Diagnosis 2: `codex --version` works in interactive shell (so PATH is fine)
# Diagnosis 3: `hermes --provider xai-oauth -m grok-4.3 -z "@file"` works
#   when invoked manually
# Diagnosis 4: `python -c "import shutil; print(shutil.which('codex'))"`
#   returns "C:\Users\RajP\AppData\Roaming\npm\codex.CMD" — extensionless shim

# Fix sequence (in order):
hermes config set providers.xai-oauth.base_url https://api.x.ai/v1
hermes config set providers.xai-oauth.model grok-4.3
# (would also need: hermes profile create xai-oauth, but Symptom 2
#  blocks before that point is reached)

# Patch council.py: -p → --provider
# Patch council.py: subprocess.run(["codex", ...]) → subprocess.run(
#     _win_exec("codex", "codex") + ["exec", ...])

# Re-run: all 3 reviewers produce non-DEGRADED output
# Codex issues BLOCK with 3 findings; council verdict becomes BLOCK (exit 2)
# — this is the correct outcome, not a regression
```

## What NOT to do

- **Don't add `shell=True` to all subprocess calls** — opens
  injection in the prompt path.
- **Don't rename the xai-oauth entry in `auth.json`** — it's correctly
  named; the problem is the missing `config.yaml` providers entry and
  the missing named profile.
- **Don't try to symlink `~/.codex/auth.json` from a different
  machine** — Codex tokens are also machine-bound.
- **Don't disable the safety-net redaction pass to "fix" a
  REDACTION_FAILED** — the artifact is being held back for a real
  reason; manually redact and re-pipe.

## Coverage gap cost

These three bugs caused a full 3-reviewer run to silently degrade to
2-of-3 in the first attempt. The user's exact request was to run a
full council; the partial run produced a PASS that was technically
true but **operationally unsafe** (Codex's missing 3 BLOCK findings
weren't surfaced). The second run, after these fixes, produced the
correct BLOCK verdict. **Worth the ~10 minutes of debug time — the
council caught real security gaps in the original artifact.**
