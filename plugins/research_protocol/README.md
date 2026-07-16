# Research Protocol plugin

PR2 exposes exactly three bounded tools in the `planner` toolset:

- `plan_context_read`: executes a closed PostgreSQL query identifier through a read-only transaction with row, byte, statement, and wall-clock caps.
- `plan_artifact_write`: validates and atomically persists a registered artifact type under an operator-owned root. Callers cannot provide a path.
- `plan_approval_request`: reloads a plan by artifact ID and expected SHA-256, checks exact scope/plan agreement, requests one-operation native Hermes consent, and persists the result.

The plugin remains disabled unless its exact public key is present in `plugins.enabled`. Registration never starts a service, opens a database connection, or performs network I/O.

## Installation

Install the optional database dependency when database-backed tools are needed:

```bash
pip install 'hermes-agent[research-protocol]'
```

## Configuration

Use an absolute, private artifact directory. Plugin settings live under `plugins.entries.research-protocol`:

```yaml
plugins:
  enabled:
    - research-protocol
  entries:
    research-protocol:
      artifact_root: /absolute/private/research-protocol
      database:
        max_rows: 100
        max_bytes: 1048576
        timeout_seconds: 5.0
        pool_min_size: 1
        pool_max_size: 4
```

Database credentials are accepted only from two fixed environment variables:

```bash
export RESEARCH_PROTOCOL_READER_DATABASE_URL='[REDACTED]'
export RESEARCH_PROTOCOL_WRITER_DATABASE_URL='[REDACTED]'
```

The reader and writer DSN strings must be distinct. They are never accepted in a tool call and are redacted from runtime representations and tool errors.

Availability is fail-closed and independent:

- `artifact_root` only: `plan_artifact_write` is available;
- reader DSN: `plan_context_read` is additionally available;
- writer DSN: `plan_approval_request` is additionally available;
- missing or malformed authority: the affected tool remains registered but unavailable through its `check_fn`.

## PostgreSQL bootstrap

Apply migrations in lexical order with an administrative migration identity:

```bash
psql -v ON_ERROR_STOP=1 "$ADMIN_DATABASE_URL" \
  -f plugins/research_protocol/migrations/0001_approval_ledger.sql
psql -v ON_ERROR_STOP=1 "$ADMIN_DATABASE_URL" \
  -f plugins/research_protocol/migrations/0002_roles.sql
```

`0001_approval_ledger.sql` and `0002_roles.sql` are intentionally one-shot.
Pre-existing protocol objects or fixed runtime role names make the transaction
fail rather than silently accepting an unknown schema or cross-database
privileges. Apply both only to a fresh installation; never rerun them as a
repair mechanism.

`0002_roles.sql` creates three NOLOGIN group roles:

- `research_protocol_planner_reader`: schema usage and table SELECT only;
- `research_protocol_approval_writer`: schema usage plus EXECUTE on the fixed `store_approval` function, with no direct table privileges;
- `research_protocol_approval_claimant`: schema usage plus EXECUTE on the fixed atomic claim function, with no direct table privileges; it is not used by the PR2 planner runtime.

Grant `research_protocol_planner_reader` and `research_protocol_approval_writer` to different LOGIN identities and put their credentials in the fixed reader/writer environment variables. Reserve `research_protocol_approval_claimant` for a future executor identity. Do not give runtime identities ownership, schema-creation rights, role administration, or migration authority.

## Security boundaries

- All input schemas are closed (`additionalProperties: false`) and expose no free path, SQL, DSN, command, or provider selector. Source URLs can occur only as validated artifact data and are never fetched by these tools.
- Context SQL comes only from the in-code registry and uses bound parameters.
- The writer credential is the trusted approval-mint authority. It is available only to the approval service after native consent, never to planner/model inputs, and can execute only the bounded `store_approval` function.
- Approval claims bind `approval_id`, canonical scope hash, plan hash, expiry, verdict, and execution count inside one fixed `SECURITY DEFINER` function; the claimant role has no direct table access.
- YOLO and `approvals.mode: off` do not bypass the native per-call elicitation used by this plugin.
- Artifact reads used for approval require the expected SHA-256 and reject symlink substitution, overwrite races, and receipt/content mismatch.

## Tests

The local suite does not require PostgreSQL:

```bash
python -m pytest -q tests/plugins/research_protocol \
  --ignore=tests/plugins/research_protocol/test_postgres_integration.py
```

The real PostgreSQL gate is opt-in and destructive only inside the database named by the test DSN:

```bash
RESEARCH_PROTOCOL_TEST_DATABASE_URL='[REDACTED]' \
  python -m pytest -q tests/plugins/research_protocol/test_postgres_integration.py
```

Use a fresh disposable database. The gate applies both migrations, truncates the approval table, and exercises replay, expiry, scope mismatch, concurrent claims, read-only transactions, caps, and timeout behavior.
