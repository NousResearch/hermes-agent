# Public Skills Over MCP interoperability fixture

A standalone synthetic SEP-2640 server and a bounded runner for a Hermes source candidate. The fixture imports only the Python MCP SDK; it does not import or copy a product server implementation.

## Run

```sh
/path/to/python run_interop.py --hermes /path/to/hermes-agent
```

The Python interpreter must have MCP 2.x and the candidate's dependencies installed. `--hermes` is a runtime path and is intentionally not embedded in fixture source. The runner creates a fresh temporary `HERMES_HOME`, starts the stdio child through Hermes, shuts it down in `finally`, and writes a receipt under `outputs/`.

The candidate must implement the SEP-2640 host integration and Skills Guard exposure checks. An unchanged Hermes revision without Skills Over MCP support is expected to fail the prerequisite check rather than receive monkeypatches or copied implementation code.

The binary control is an actual 1×1 PNG. It must remain labeled `not_scanned_binary`; its successful materialization is not a content-safety verdict. Arbitrary `.bin` files remain subject to the existing Skills Guard structural policy.

`tests/tools/test_mcp_skills_interop.py` exposes separate native `linux_only` and `macos_only` tests for the repository's CI discovery. The opposite-host test is skipped honestly. `shutdown_returned` records only that the bounded host shutdown API returned without exception, not an independent assertion that every child process was reaped.

See `HOST_CONSTRAINTS.md` for the exact covered and uncovered modes.
