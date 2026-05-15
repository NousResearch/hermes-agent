# mcporter secret placeholders

Hermes' mcporter bridge resolves secret placeholders before invoking `mcporter`.
This keeps real tokens out of `config/mcporter.json` while still giving mcporter
an ordinary resolved config file at call time.

Supported placeholders inside mcporter JSON string values:

- `${env:NAME}` — read directly from the Hermes process environment.
- `${vault:path/to/secret}` — read via Hermes' vault resolver. Until a first-class
  in-process vault API is available, this maps to a namespaced environment
  variable: `HERMES_VAULT_PATH_TO_SECRET` with non-alphanumeric characters
  converted to underscores and uppercased.

Example:

```json
{
  "headers": {
    "Authorization": "Bearer ${vault:prod/hermes/zai_mcp_token}"
  }
}
```

Fallback env variable for that example:

```bash
export HERMES_VAULT_PROD_HERMES_ZAI_MCP_TOKEN=...
```

Runtime behavior:

1. Hermes reads the configured mcporter JSON.
2. If placeholders are present, Hermes resolves them recursively.
3. Hermes writes a temporary resolved config file in a `0700` temp directory with
   file mode `0600`.
4. Hermes invokes mcporter with `--config <temp-file>`.
5. Hermes deletes the temp file and directory after mcporter exits.

Safety rules:

- unresolved placeholders block the mcporter call
- the original config path is used unchanged when no supported placeholders exist
- resolved configs are never logged or returned to the model
- mcporter stderr/stdout errors still pass through Hermes redaction
- commit examples only; never commit real `Authorization` values
