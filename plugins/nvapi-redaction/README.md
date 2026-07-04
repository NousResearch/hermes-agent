# nvapi-redaction

Masks NVIDIA API keys (`nvapi-...`, used by NIM endpoints and
build.nvidia.com) in logs, terminal output, transport errors, and
transcripts — everywhere the built-in vendor prefixes are masked.

Also the reference implementation for the
`ctx.register_redaction_patterns()` plugin interface: vendor token
formats as plugins instead of one-line core PRs to
`agent/redact.py::_PREFIX_PATTERNS`.

## Enable

```bash
hermes plugins enable nvapi-redaction
```

Respects the global `security.redact_secrets` setting like every
built-in pattern. Additive-only: this plugin (and any redaction plugin)
can extend masking but cannot weaken it.
