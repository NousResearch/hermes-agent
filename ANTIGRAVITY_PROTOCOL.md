# Antigravity CLI structured-protocol spike

## Scope and evidence

This is an empirical record, not a compatibility promise. The requested binary
`/home/rafalmuraro/.local/bin/agy` reported **1.2.7** on 2026-09-19, rather
than the requested 1.2.5. Every protocol finding below therefore applies only
to the installed 1.2.7 binary.

The reproducible probe is `scripts/antigravity_probe.py`. It creates a fresh
temporary directory, uses `--sandbox`, never passes
`--dangerously-skip-permissions`, asks no agent to edit files, and removes the
temporary directory when finished.

```sh
cd /home/rafalmuraro/.hermes/hermes-agent
python3 scripts/antigravity_probe.py \
  --agy /home/rafalmuraro/.local/bin/agy \
  --output /tmp/antigravity-protocol-report.json
```

The report stores raw stdout/stderr, exact argv, exit status, and a parsed
summary for each experiment. Its output is deliberately outside the repository
because it contains ephemeral conversation IDs and temporary paths.

## Invocation and input

`agy --help` describes `stream-json` as one NDJSON message per stdin line and
requires `--output-format stream-json`. The observed working invocation is:

```sh
printf '%s\n' \
  '{"event":"user","message":{"role":"user","content":"Reply with exactly PONG. Do not use tools."}}' |
/home/rafalmuraro/.local/bin/agy \
  --input-format stream-json --output-format stream-json \
  --sandbox --print='' --print-timeout 60s
```

Observed accepted input shape:

```json
{"event":"user","message":{"role":"user","content":"..."}}
```

The empty `--print=''` was required in this observed non-interactive stdin
mode. `--print-timeout` requires a duration suffix; the probe uses `60s`.

Malformed input (`not json`) exited 1, wrote an `error:` diagnostic on stderr,
and still emitted two NDJSON records: `init`, then `result` with
`result.status: "ERROR"`, `num_turns: 0`, and the decode error in
`result.error`.

## Output envelope and turn records

A successful one-turn input emitted NDJSON outer events, in this order:

1. `init`
2. `step_update` for `user_input`, `state: "DONE"`, `step_index: 0`
3. `step_update` for `agent_response`, `state: "DONE"`, `step_index: 1`
4. `result`, `status: "SUCCESS"`, `num_turns: 1`

The `init` event included the `conversation_id` plus
`init.cwd`, `init.tools`, and `init.permission_mode: "request-review"`.
Every observed `step_update` and `result` also carried the conversation ID.
`agent_response` carried `text_delta`, `duration_seconds`, and `usage`;
`result` carried `response`, `duration_seconds`, `num_turns`, and cumulative
`usage`.

Observed outer event names: `init`, `step_update`, `result`.

Observed step types:

- `user_input`
- `agent_response`
- `system_message` (during successful resume)
- `tool` (permission experiment)

No other event or step types are asserted by this spike.

## Conversation resume

The probe first obtains an ID from a successful run, then invokes:

```sh
/home/rafalmuraro/.local/bin/agy ... --conversation <returned-id>
```

The resumed run retained the same ID. Its observed step indexes continued at
2: `user_input` (2), `system_message` (3), and `agent_response` (4). The final
result had `num_turns: 2`. This establishes only resume-by-returned-ID; unknown
or expired ID behavior was not included in the reproducible probe.

## Tool and permission behavior

The init record advertised 57 tool names in the probe environment, including
file operations, browser operations, `run_command`, `search_web`, MCP,
subagent, scheduling, and permission tools. This is advertised capability, not
proof that each is usable.

A read-only `run_command` request for `pwd` produced:

```json
{"step_type":"tool","state":"ACTIVE","tool_name":"run_command"}
{"step_type":"tool","state":"ERROR","tool_name":"run_command","tool_info":{"error":{"type":"TOOL_ERROR"}}}
```

In headless print mode it was auto-denied rather than executed. stderr said the
command permission could not be prompted for; the final `result` nevertheless
had `status: "SUCCESS"`, an empty response, and
`denied_actions: [{"action":"command","display_name":"RunCommand"}]`.
The probe does not bypass this denial.

The official headless-mode documentation confirms this is a protocol boundary,
not an adapter omission: print/headless mode has no interactive permission prompt.
Actions requiring approval are soft-denied unless a rule already exists in the
Antigravity settings; the only global CLI override is
`--dangerously-skip-permissions`. Hermes deliberately does not use that override
or mutate the user's Antigravity permission policy. Consequently the V1 adapter
fails the turn loudly when `denied_actions` is present, but a Hermes Desktop
**allow once / deny** round-trip cannot be implemented on the documented
`stream-json` interface in agy 1.2.7. Interactive approvals exist in the
Antigravity TUI/Remote Control surfaces, but no local machine-readable approval
request/response contract is documented for third-party control planes.

Reference: <https://antigravity.google/docs/cli/headless/#permissions-in-headless-mode>

## Timeout / cancellation boundary

With `--print-timeout 1ms`, a turn in progress wrote this stderr diagnostic:

```text
[agy] print timeout after 1ms with turn in progress; returning partial output
```

It exited 0 and emitted `init` then `result` with `status: "SUCCESS"`, empty
response, and `num_turns: 0`. This demonstrates the print-timeout partial-return
path. It does **not** demonstrate signal-based cancellation or server-side
cancellation semantics.

## Observable configuration/authentication state

- `agy models` completed and listed 14 models, demonstrating that this local
  invocation could reach its model catalog and execute prompts. Hermes exposes
  those IDs dynamically in the model picker and passes an explicit selection as
  `--model <id>`; selecting `auto` omits the flag and leaves routing to `agy`.
- `agy agents` exited 0 with no output.
- `agy mcp list` returned `No MCP servers configured.`
- `agy plugin list` returned `No imported plugins.`
- `agy --help` listed no authentication subcommand. The authentication mechanism
  and behavior without existing credentials were not tested.

## Test coverage

`tests/test_antigravity_probe.py` exercises the pure NDJSON parser and summary
contract without invoking the CLI. Run it with:

```sh
uv run --with pytest pytest tests/test_antigravity_probe.py -q
```
