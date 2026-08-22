# Secret-masking corrupts file writes — literal `***` lands in scripts

Observed 2026-08-03 (LAH_MCP_BRIDGE_PERMANENT_READONLY_OBSERVABILITY_V1), costing
several repair iterations. This is a Hermes tool-layer behavior, NOT a shell issue.

## Symptom

Writing a shell/Python file that contains a credential-assignment pattern such as:

```bash
AUTH="$(printf '%s:%s' "$WP_APP_USERNAME" "$WP_APP_PASSWORD" | base64 -w0)"
```

via `write_file`, `patch`, or `execute_code` can land the **literal bytes**
`AUTH=*** ...` in the file on disk — not just in the terminal echo. The script then
silently misbehaves or fails.

Two distinct consequences:

1. **File corruption**: `grep -n AUTH file` and `cat file` show the masked form, and
   the FILE ITSELF contains `***`. `bash -n` may still PASS if the corruption happens
   to leave syntactically valid (but wrong) content — so syntax check is NOT proof.
2. **Patch tool "identical" errors**: when the masked region is the only difference
   between `old_string` and `new_string`, `patch` reports "old_string and new_string
   are identical" and refuses, because both sides were masked in the tool layer.

## Verification that actually works

Terminal echo of the file is ALSO masked, so `cat`/`grep` output cannot be trusted to
prove the bytes. Only byte-level checks are conclusive:

```python
# via execute_code (reads the real bytes):
with open(path, 'rb') as f:
    data = f.read()
assert b'***' not in data, "file contains masked literal ***"
```

Or compare SHA-256 of the on-disk file against SHA-256 of the intended content built
in the same execute_code run — mismatch means the write was corrupted.

## Avoidance — construct files so the masked pattern never appears

- **curl native `-u "$USER:$PASS"`**: curl base64-encodes Basic auth internally, so
  no `AUTH="$(...)` variable is needed at all. This is the cleanest fix.
- **`docker exec <container> printenv VAR` for runtime secrets** (production smoke
  tests, observed 2026-08-05 PR #701): instead of `grep '^VAR=' env | cut` or a
  `VAR="$(...)"` assignment line, read the secret straight into the shell with
  `K=$(docker exec <container> printenv VAR)` then use it inline. `printenv` output
  is not a credential-assignment pattern, so the masker does not rewrite it. Verify
  length only: `echo "loaded: ${#K} chars"` — never print the value.
- **Inline curl `-H "header: ${VAR}"` survives; a `AUTH="...${VAR}"` assignment
  line gets corrupted** (observed 2026-08-05): `probe-ds.sh` with the header
  written directly inside `curl -H "authorization: Bearer ${DSK}"` ran correctly,
  while an intermediate `AUTH="authorization: Bearer ${DSK}"` variable line landed
  on disk as `AUTH=*** ..."` (broken quote → `unexpected EOF while looking for
  matching quote`). Prefer putting the secret-bearing header inline in the command
  that consumes it; if you must assign, build the line from concatenated pieces
  (`'authorization' + ': Bearer ' + key`).
- **grep/cut extraction instead of `source` + variable build**:
  ```bash
  WP_APP_USERNAME="$(grep -E '^export WP_APP_USERNAME=' "$SECRETS_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
  WP_APP_PASSWORD="$(grep -E '^export WP_APP_PASSWORD=' "$SECRETS_FILE" | head -1 | cut -d= -f2- | tr -d '"' | tr -d "'")"
  ```
  This pattern (grep/cut + read-from-file) is the known-safe form; the masker does not
  rewrite it.
- **Python helper reads the env file directly** and prints the needed value; the
  script then calls the helper instead of containing a credential assignment.

## JS/Node test-fixture variant (write_file corruption)

Observed 2026-08-05 (CLOE_CANONICAL_EXACT_SHA_DEPLOYMENT_CAPABILITY_V1). The masker
also corrupts JS test files written via `write_file` when a string literal matches a
credential-assignment shape:

```js
writeFileSync(source, `ADMIN_API_KEY=${secr...
// lands on disk as:  writeFileSync(source, `ADMIN_API_KEY=***  → SyntaxError
```

Symptoms that cost real debugging time:
- `writeFileSync(source, 'ADMIN_API_KEY=placeh....')` came back truncated to
  `'ADMIN_API_KEY=***` — syntactically VALID but WRONG content, so the test silently
  asserted against a mangled string and failed later for a confusing reason (e.g. a
  mock fs mock returning a truncated value → "value not found" downstream).
- Template literals embedding a `KEY=` prefix get truncated mid-expression, producing
  a SyntaxError whose line number points at a different root cause.
- Affects `write_file`, `patch`, and inline `node -e`/`python3 -c` strings alike: the
  masking happens in the tool layer before bytes reach disk/shell.

Avoidance for JS fixtures (same principle as the shell forms above):
- **Concatenate the key name** so the literal never appears:
  ```js
  const envLine = 'ADMIN_API_KEY' + '=' + secretValue;
  writeFileSync(source, envLine + '\nGIT_COMMIT=stale-value\n');
  ```
- **Use a non-secret-looking placeholder** (`DB_CONN_STRING=benign-placeholder-value`)
  for generic fs mocks that don't specifically test redaction.
- Verify with `node --check` AND read the actual on-disk bytes (assert the expected
  substring via readFileSync) — echo is masked, only byte-level checks are conclusive.

## JS test env-assignment variant (process.env.* = 'value')

Observed 2026-08-08 (CLOE_LAHB_AUTONOMOUS_AFFILIATE_RUNTIME_E2E_V1, Lots 2–3). The
masker mangles **test-file env setup lines** when the assignment value looks like a
credential, even a throwaway test value:

```js
// written via write_file lands on disk as:
process.env.LAHB_ADMIN_API_KEY='***';          // ← literal ***, syntax still OK but WRONG value
if (prevKey === undefined) delete process.env.LAHB_ADMIN_API_KEY; else process.env.LAHB_ADMIN_API_KEY=***  // ← SyntaxError
```

Distinct symptoms vs the fixture variant:
- The `'***'` replacement keeps the line syntactically valid (value replaced), so
  `node --check` PASSES and the test fails later with a confusing `LAHB_ADMIN_API_KEY_REQUIRED`
  or `Unauthorized` — the bridge never sees the test key.
- `patch` on the mutilated line can compound the damage (dropped `}` closing a
  try/finally → cascading SyntaxErrors at EOF; LSP diagnostics point at the last
  `});`). When this happens, re-write the WHOLE file rather than patching line-by-line.
- Affects both the assignment AND the `else process.env.X = prevKey` teardown line —
  teardown is the one that produces the hard SyntaxError.

Reliable avoidance (used successfully Lots 2–3) — never write the literal env name
or a secret-looking value; build both from concatenation and centralize in helpers:

```js
const KEY_ENV = 'LAHB_' + 'ADMIN_API_KEY';   // bracket access — no literal KEY= assignment
const KEY_VALUE = 'lahb-' + 'test-' + 'key'; // benign-looking, never 'secret'
const URL_ENV = 'LAHB_' + 'URL';

function setupEnv() {
  const prevUrl = process.env[URL_ENV];
  const prevKey = process.env[KEY_ENV];
  process.env[URL_ENV] = 'https://leanframeworklab.com';
  process.env[KEY_ENV] = KEY_VALUE;
  return { prevUrl, prevKey };
}
function teardownEnv(prevUrl, prevKey) {
  if (prevUrl === undefined) delete process.env[URL_ENV]; else process.env[URL_ENV] = prevUrl;
  if (prevKey === undefined) delete process.env[KEY_ENV]; else process.env[KEY_ENV] = prevKey;
}
// each test: const { prevUrl, prevKey } = setupEnv(); try { ... } finally { teardownEnv(prevUrl, prevKey); }
```

This also removes the per-test try/finally boilerplate that the masker keeps breaking.
Do NOT try to write the env name inside a write_file script in any form — the
masking layer rewrites the bytes as the file is written, so the only safe path is
concatenation + bracket notation.

## Rule of thumb

Any time a file you just wrote contains a variable assignment whose value is derived
from a secret (`PASSWORD`, `TOKEN`, `AUTH`, `KEY`, `SECRET`), verify the bytes with
`b'***' not in data` before trusting it — and prefer the `-u`/grep-cut forms above
so the pattern never enters the file in the first place.
