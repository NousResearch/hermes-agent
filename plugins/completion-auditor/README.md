# completion-auditor

`completion-auditor` is a disabled-by-default, audit-only Hermes plugin that records local JSONL metadata about whether an assistant's final completion claim is supported by same-turn tool evidence.

It does **not** block, repair, rewrite, or judge the user-visible final response.

## Enable

Add the plugin id to Hermes config:

```yaml
plugins:
  enabled:
    - completion-auditor
  completion_auditor:
    mode: audit
    log_verdicts: true
```

The runtime config key may also be written as `completion-auditor`, but `completion_auditor` is the documented form.

Restart Hermes after changing plugin config.

## Logs

Default log directory:

```text
~/.hermes/logs/completion-auditor/
```

Records are JSONL and use schema:

```text
hermes-completion-audit-v1
```

Files are created with private permissions where POSIX mode bits apply:

- directory: `0700`
- log files: `0600`

## Configuration

```yaml
plugins:
  completion_auditor:
    mode: audit
    log_verdicts: true
    log_dir: ~/.hermes/logs/completion-auditor
    include_tool_result_excerpt: false
    max_result_excerpt_chars: 800
    redact_secrets: true
    log_retention_days: 7
    max_log_size_mb: 10
```

Notes:

- `mode` supports only `audit` in the MVP. Other modes are treated as disabled.
- Tool result excerpts are off by default.
- If excerpts are enabled, common secret shapes are redacted before writing.
- Logs rotate by size and prune old `completion-audit-*.jsonl*` files by modification time.

## Verdicts

- `supported`: the specific claim is supported by same-turn direct/strong evidence.
- `weak`: a claim exists, but evidence is missing, indirect, ambiguous, or scope-mismatched.
- `fail`: direct evidence contradicts the claim.
- `not_applicable`: no checkable completion claim was detected.
- `audit_error`: the auditor could not run correctly, such as missing turn/session identity.

## Evidence tiers

- `tier_1`: direct structured evidence for the claim, such as `exit_code=0` for a test claim or matching `patch`/`write_file` path for a file-modification claim.
- `tier_2`: strong but indirect evidence.
- `tier_3`: weak contextual evidence.
- `tier_4`: missing or unusable evidence.

## What this does not prove

`completion-auditor` does not guarantee semantic task correctness, full user-goal completion, or absence of hidden bugs. `supported` only means the available Hermes hook metadata supports the specific final-response claim.

The plugin is intentionally local-only and deterministic by default. It does not run verifier commands or call an LLM judge.
