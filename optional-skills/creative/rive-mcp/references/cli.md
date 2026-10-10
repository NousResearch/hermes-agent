# Guarded MCP helper (not the official Rive CLI)

Prefer native Hermes MCP. This optional Python client supports the same local
editor when native tools are unavailable; it is not a compiler, an MCP server,
a service, or the third-party RiveMCP product. For source-controlled RML builds,
read `official-cli.md` instead.

## Isolated installation

Use `terminal` with Python 3.11+ to create an ordinary task-owned virtual
environment. Install only after authorization; never change Hermes's own runtime:

```text
python -m venv .rive-helper-venv
```

On macOS/Linux, use `.rive-helper-venv/bin/python`; on Windows use
`.rive-helper-venv\Scripts\python.exe`. With that interpreter, use `terminal`:

```text
python -m pip install -r "{SKILL_DIR}/scripts/requirements.txt"
python "{SKILL_DIR}/scripts/rive_mcp.py" --help
python "{SKILL_DIR}/scripts/rive_doctor.py" --json
python "{SKILL_DIR}/scripts/rive_doctor.py" --handshake --json
```

Resolve `{SKILL_DIR}` via `skill_view`; substitute the environment's interpreter
for `python`. The requirements are bounded for MCP SDK 2.x (`httpx2`, not the older
SDK's `httpx` transport). Record resolved versions for reproduction. Help and the
TCP-only doctor do not require those dependencies; other commands fail explicitly
if they are missing. The scripts do not install or launch anything.

## Read and preflight

Run these through `terminal`:

```text
python "{SKILL_DIR}/scripts/rive_mcp.py" list
python "{SKILL_DIR}/scripts/rive_mcp.py" schema session_info
python "{SKILL_DIR}/scripts/rive_mcp.py" call session_info --args "{SKILL_DIR}/scripts/empty-args.json"
python "{SKILL_DIR}/scripts/rive_mcp.py" call open_file_editor --args "{SKILL_DIR}/scripts/current-file-args.json"
python "{SKILL_DIR}/scripts/rive_mcp.py" call list_artboards --args "{SKILL_DIR}/scripts/empty-args.json" --expected-file FILE_ID
python "{SKILL_DIR}/scripts/rive_mcp.py" call TOOL --args arguments.json --dry-run
```

Use `write_file` to create an argument file from the **live** schema. Do not place
credentials in command-line arguments. Strict JSON rejects duplicate keys, NaN,
infinities, overflow and non-object roots. Discovery follows pagination and
rejects duplicate tool names, cursor loops and excessive pages.

`--dry-run` connects, initializes, discovers and validates the live schema and
local policy. It calls **no** tool, not even `session_info`. It does not establish
file identity, object existence, permission, valid graph wiring or successful edits.

## Apply a bounded change

After the user has authorized the actual change, call through `terminal`:

```text
python "{SKILL_DIR}/scripts/rive_mcp.py" call TOOL --args arguments.json --apply --expected-file FILE_ID
```

`--apply` is a deliberate flag, not a substitute for user authorization. Unknown
and mutation-class calls require both flags. `session_info` must return one
consistent, nonempty ID matching `FILE_ID` immediately before the call. This
preflight is not an atomic lock: one controller, no concurrent tab switches,
then exact-target readback. `--expected-file` can also scope reviewed reads.

The read policy is an explicit allowlist of tools, commands and argument keys.
Tool names and `readOnlyHint` alone confer no trust. `capture_artboard` remains
mutation-class pending a separate execution/safety review; discovery of the tool
does not loosen the guard. Script tests and state simulation can execute code.

## Network, errors and privacy

The helper fixes the endpoint to `http://127.0.0.1:9791/mcp`, ignores proxy
variables, refuses redirects and external schema references, and disables
transport retries. No remote endpoint/auth flags are provided. A local endpoint
does not make cloud-backed Rive documents or the model provider offline.

It detects nested MCP/application error envelopes. Diagnostics intentionally
withhold raw argument/server text; successful results can still contain private
document content, so do not publish them indiscriminately. No telemetry is added.
Never automatically retry an uncertain mutation; inspect the editor first.

Exit codes:

| Program | Code | Meaning |
| --- | --- | --- |
| Helper | 0 | Requested schema/list/call succeeded |
| Helper | 2 | Arguments, dependency, policy, schema or reported server error |
| Helper | 3 | Transport/protocol failure; mutation outcome may be unknown |
| Doctor | 0 | Explicit SDK handshake and catalog verified |
| Doctor | 1 | Endpoint closed or handshake failed |
| Doctor | 2 | TCP listener only; MCP not verified |

The doctor's `--handshake` initializes and lists tools but never calls a tool.
Neither doctor mode proves an open file, authoring, capture, export or entitlement.

## Verification

From `scripts/`, use the isolated interpreter through `terminal`:

```text
python -B -m unittest -v test_rive_mcp
```

The regression suite covers guards and real SDK behavior against synthetic
transports. It does not require Rive, access the editor or claim live authoring.
In the Hermes repository, use its required runner for
`tests/skills/test_rive_mcp_optional_skill.py`. Independently verify a real
`session_info` call when a live editor connection is part of the requested task.
