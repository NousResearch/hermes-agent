# DCG Mode for Hermes

When `approvals.mode: dcg` (or `approvals.dcg.enabled: true`) is set in
`~/.hermes/config.yaml`, Hermes delegates **all** terminal command safety
decisions to the external `dcg` binary
(https://github.com/Dicklesworthstone/destructive_command_guard).

## Philosophy

> "Destructive command guard should block the worst; everything else generally
> is runnable." — user, 2026-04

This replaces Hermes's own pattern-matching + tirith prompt pipeline with a
single higher-signal gate. The result:

- Genuinely destructive commands (`rm -rf /`, `dd` to raw devices, git history
  rewrites, fork bombs, etc.) are blocked by dcg's curated rule packs.
- Benign-but-flagged commands (`python3 -c "print(1)"`, `bash -lc "..."`) that
  previously triggered Hermes approval prompts now run without interruption.

## Enabling

In `~/.hermes/config.yaml`:

```yaml
approvals:
  mode: dcg
  timeout: 60
  dcg:
    enabled: true
    path: dcg         # or absolute path
    timeout: 3        # seconds
```

You can also set `HERMES_DCG_MODE=1` for a one-shot override.

## Behavior

- dcg `allow` (or empty stdout): Hermes proceeds without prompting.
- dcg `deny`: Hermes returns `BLOCKED by dcg (<rule_id>): ...` to the agent
  with an explicit "do not retry" directive. The user is *not* prompted —
  honors the user preference that dcg's word is final. If the user wants to
  override, they allowlist the rule via `dcg allow <pack>:<rule> --project`
  or run the command manually.
- dcg missing / errors / timeouts: soft-allow. We fail open to avoid
  lockouts, and log a warning.

## Layered guards (off by default)

Hermes's legacy tirith + dangerous-command detection remain available when
`approvals.mode` is `manual`, `smart`, or `off`. Choose dcg-mode if you want
a single high-signal gate and minimal interruption.

## Files

- `tools/dcg_guard.py` — dcg subprocess wrapper + config helpers.
- `tools/approval.py` — `check_all_command_guards()` short-circuits to dcg
  when dcg-mode is enabled, bypassing tirith + pattern matching.
