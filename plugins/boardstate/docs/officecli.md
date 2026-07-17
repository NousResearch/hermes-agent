# OfficeCLI connector (Office Ops)

OfficeCLI ([github.com/iOfficeAI/OfficeCLI](https://github.com/iOfficeAI/OfficeCLI),
Apache-2.0) ships an MCP server — `officecli mcp` speaks MCP over stdio — so it is the first
blessed Boardstate connector: no cloud auth, and it demos `.docx` / `.xlsx` artifacts. This
plugin **bundles nothing** — it detects the binary and instructs; you install and author the
config.

## 1. Install the binary (detect-or-instruct)

Install OfficeCLI so `officecli` is on your `PATH`:

```sh
brew install officecli   # or a GitHub release
```

On first boot with no connectors configured, the sidecar logs whether `officecli` was found
and prints the exact config to author (below).

## 2. Author the connector (the authorship boundary)

A connector exists **only** because you named it in `boardstate.connectors.json` in the
Boardstate state dir (`$HERMES_HOME/boardstate-state/boardstate.connectors.json`, or the
`BOARDSTATE_HERMES_STATE_DIR` override). It is **never** read from the board document.

```json
{
  "connectors": [
    { "name": "officecli", "transport": "stdio", "command": "officecli", "args": ["mcp"] }
  ]
}
```

Keep the file owner-only (the sidecar `chmod 600`s it on load). OfficeCLI needs no secrets —
no `env` refs, no headers.

## 3. Approve tools (nothing runs until you do)

Restart the dashboard. The agent can now **discover** the connector's tools
(`boardstate_tool_search`) and **request** grants, but it cannot grant them. Open the
approvals panel and approve the specific tools you want. A **read-only** tool then executes
directly; a **mutating** tool **parks for your confirm** before it runs.

Approve/confirm decisions travel the operator gate only
(`POST /api/plugins/boardstate/operator`), never the browser WebSocket or the agent's MCP
connection.

### Auto-run (autoConfirm) — an operator opt-in, off by default

The approvals panel's per-tool **Auto-run** checkbox sets `autoConfirm` on a grant: an
auto-confirmed tool executes **without** parking a pending action for per-call confirmation.
It is an operator opt-in and is **off by default** — the plugin never enables it, and it is
not a `boardstate.connectors.json` field (the config parser rejects it). Only enable Auto-run
for fully-trusted, low-consequence tools; leave it off so every mutating call waits for your
explicit confirm.

## 4. Operate

Apply the **Office Ops** board template (the Templates row on the Board tab): a grant-gated
action button that generates a document, plus an artifacts note. Or just ask the agent to
build a workbook / compose a report — every consequential action is operator-confirmed, and
the generated artifact is linked on the board.

### Multi-user / gated dashboards

Behind an auth gate (non-loopback bind), the operator endpoint additionally requires an
allowlist: author `boardstate.operators.json` in the state dir listing the principals
(emails) permitted to approve —

```json
{ "operators": ["you@example.com"] }
```

Absent that file in gated mode, the operator endpoint refuses every request (403): an
undifferentiated session token must not be a self-service operator. The gate also fails
**closed** when it cannot positively confirm a loopback single-user bind — if the dashboard
mode is indeterminate, the allowlist is required.

### Operator actions and the shared sidecar

One sidecar is shared per state dir: a second backend (e.g. the desktop app alongside the web
dashboard) **adopts** the running sidecar via the state-dir port file to render the board. The
operator plane is gated by a **dedicated secret** that is held only in memory by the backend
that **spawned** the sidecar — it is never written to the port file. So an *adopting* backend
can render the board but cannot drive operator approve/confirm/deny (it returns 503 with a
clear message); drive operator actions from the spawning backend, or restart so your backend
owns the sidecar. This is deliberate: knowing the port-file contents must never be enough to
approve a grant.
