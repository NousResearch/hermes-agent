# CCD MCP Integration

Wire [CCD](https://github.com/dusk-network/ccd) (Continuous Context Development) into
Hermes Agent as a native MCP server so its session-management, memory, and governance
tools are available alongside built-in agent tools.

## Prerequisites

| Requirement | How to check |
|---|---|
| CCD binary on PATH | `ccd --version` → `1.0.0-alpha` or later |
| CCD profile configured | `CCD_PROFILE=<name> ccd doctor /path/to/repo` |
| `mcp` Python package | `pip show mcp` (install with `pip install mcp`) |

## Configuration

Add the following to `~/.hermes/config.yaml` (or your profile-specific config):

```yaml
mcp_servers:
  ccd:
    command: /Users/nanto/.cargo/bin/ccd   # or just "ccd" if on PATH
    args:
      - mcp-serve
    env:
      CCD_PROFILE: raoh                    # CCD profile to use
    timeout: 120                           # per-tool-call timeout (seconds)
    connect_timeout: 30                    # initial handshake timeout (seconds)
```

On agent startup, Hermes will:

1. Spawn `ccd mcp-serve` as a stdio subprocess
2. Complete the MCP initialize handshake
3. Discover all CCD tools via `tools/list`
4. Register them with the `mcp_ccd_` prefix

## Exposed Tools

After discovery, the following tools become available (names prefixed `mcp_ccd_`):

| MCP Tool | Description |
|---|---|
| `ccd_repo` | Repo lifecycle: attach, scaffold, link, unlink, gc, skills-install |
| `ccd_health` | Validate: doctor, check, drift, sync, preflight, hooks |
| `ccd_session` | Session lifecycle: start, open, state-start, state-clear |
| `ccd_session_lifecycle` | Runtime session lifecycle: start, heartbeat, clear, takeover |
| `ccd_session_gates` | Execution gates: list, replace, seed, set-status, advance, clear |
| `ccd_escalation` | Escalation state: list, set, clear |
| `ccd_recovery` | Recovery artifacts: write checkpoint or working buffer |
| `ccd_state` | State and governance: export, radar, checkpoint, handoff, policy |
| `ccd_delegation` | Delegation bootstrap: bounded child context for sub-agents |
| `ccd_context` | Mid-session refresh evaluation: context-check |
| `ccd_backlog` | Work queue: pull, lint, groom, bootstrap |
| `ccd_memory` | Memory follow-through: candidate-admit, compact, promote |
| `ccd_memory_recall` | Memory recall: search, describe, expand |

Plus standard MCP meta-tools: `list_resources`, `read_resource`, `list_prompts`, `get_prompt`.

## Verification

```bash
# 1. Test the MCP handshake directly
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"test","version":"0.1"}}}' \
  | CCD_PROFILE=raoh ccd mcp-serve

# 2. Verify tools appear in Hermes
#    Start Hermes Agent and check startup logs for:
#    "Connected to MCP server 'ccd' (13 tools)"
```

## Troubleshooting

- **"MCP SDK not available"**: Install with `pip install mcp`
- **Connection timeout**: Increase `connect_timeout` or verify `ccd mcp-serve` starts quickly
- **Wrong profile**: Ensure `CCD_PROFILE` in the env section matches your desired CCD profile
- **Tools not appearing**: Check that `mcp_servers` is a top-level key (not nested under another key)

## References

- [Native MCP Skill](../skills/mcp/native-mcp/SKILL.md) — general MCP client docs
- [CCD MCP tools source](https://github.com/nousresearch/ccd/blob/main/src/mcp/tools.rs)
- GitHub Issue: [#4837](https://github.com/NousResearch/hermes-agent/issues/4837)
