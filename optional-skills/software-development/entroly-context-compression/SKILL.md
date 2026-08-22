---
name: entroly-context-compression
description: "Optimize coding context with Entroly's local MCP tools."
version: 1.0.0
author: juyterman1000, Hermes Agent
license: Apache-2.0
platforms: [linux, macos, windows]
prerequisites:
  commands: [python, hermes]
metadata:
  hermes:
    tags: [context-management, context-compression, mcp, coding, verification]
    category: software-development
    related_skills: [code-wiki, codebase-inspection]
    homepage: https://github.com/juyterman1000/entroly
---

# Entroly Context Compression Skill

Use Entroly as an external, optional local MCP server when a coding task needs bounded
context selection, inspectable receipts, exact recovery data, or a verification
signal. This integration does not replace Hermes's built-in `ContextCompressor`
or `ContextEngine`, and it does not intercept model-provider traffic.

## When to Use

Use this skill when:

- a repository is too large to send as unfiltered context;
- the user asks for a token budget, selected-source explanation, or receipt;
- omitted evidence may need to be recovered exactly later; or
- a draft answer should be checked against the context that was supplied.

Do not use it for a small, direct file lookup, when the user requires the
unaltered source, or as a substitute for Hermes's automatic conversation
compaction. Prefer Hermes's built-in compressor for normal context-window
management.

## Prerequisites

- Hermes Agent with MCP support.
- Python 3.10 or newer for the Entroly package.
- A local stdio MCP server named `entroly` registered with Hermes; the next
  section creates and hardens this entry.
- A terminal opened at the repository root that Entroly should inspect.
- Permission to create local Entroly state in that workspace.

Install the base package; the MCP server does not require the proxy extra:

```bash
python -m pip install -U entroly
```

No model-provider API key is required for the local MCP operations in this
skill. If a transparent OpenAI-compatible proxy is separately required, install
`entroly[proxy]`; that mode is outside this MCP workflow and is not enabled by
installing this skill.

## How to Run

From the repository root, use Hermes's `terminal` tool to install this optional
skill and register Entroly as a stdio MCP server:

```bash
hermes skills install official/software-development/entroly-context-compression
hermes mcp add entroly --command entroly
```

The add command probes the server before saving it. At the tool-selection
prompt, choose `select` and enable only the tools listed in the configuration
below. The add flow stores that native-tool allowlist, but it leaves Hermes's
MCP resource and prompt utility wrappers enabled by default. Before starting a
session, open the active profile configuration:

```bash
hermes config edit
```

Under the existing top-level `mcp_servers:` key, update only the generated
`entroly` child so it matches this hardened mapping. Preserve every other MCP
server entry:

```yaml
mcp_servers:
  entroly:
    command: "entroly"
    args: []
    enabled: true
    connect_timeout: 60
    timeout: 120
    supports_parallel_tool_calls: false
    tools:
      include:
        - smart_read
        - remember_fragment
        - optimize_context
        - explain_context
        - entroly_retrieve
        - recall_relevant
        - create_context_receipt
        - create_context_receipt_from_path
        - render_context_receipt
        - explain_receipt_omission
        - recover_receipt_omission
        - verify_response
      resources: false
      prompts: false
```

Save the file, then verify the final connection:

```bash
hermes mcp test entroly
hermes mcp list
```

Start a new Hermes session after registration. In an existing chat, run
`/reload-mcp` and approve the reload when prompted.

The allowlist uses Entroly's server-native names. In chat, Hermes exposes them
with an `mcp_entroly_` prefix, such as `mcp_entroly_smart_read`.

Ask for an explicit, observable operation rather than automatic compression:

> Use Entroly's MCP tools to select context for this task under 8,000 tokens.
> Show the selected sources, warnings, and recovery handles. Do not treat the
> selection or verification score as proof.

## Quick Reference

| Goal | Command or tool |
| --- | --- |
| Install Entroly MCP support | `python -m pip install -U entroly` |
| Register the stdio server | `hermes mcp add entroly --command entroly` |
| Harden the active profile | `hermes config edit` |
| Test the connection | `hermes mcp test entroly` |
| Show registered servers | `hermes mcp list` |
| Refresh a running chat | `/reload-mcp` |
| Remove the server | `hermes mcp remove entroly` |
| Read one file to a budget | `mcp_entroly_smart_read` |
| Select stored context | `mcp_entroly_optimize_context` |
| Inspect the last selection | `mcp_entroly_explain_context` |
| Build a recoverable receipt | `mcp_entroly_create_context_receipt` with `recoverable=true` |
| Check a draft against context | `mcp_entroly_verify_response` |

## Procedure

1. **Choose the right context path.** Keep Hermes's built-in compressor and
   `ContextEngine` enabled for conversation compaction. Use Entroly only for an
   explicit context-selection, receipt, recovery, or verification operation.
2. **Start at the intended project root.** Entroly scopes relative file access
   and local state to the server's workspace. Stop and restart Hermes from the
   correct directory if the wrong repository is visible.
3. **Install, register, and harden.** Run the installation and `hermes mcp add`
   commands above. Select the narrow native-tool allowlist, then use
   `hermes config edit` to apply the complete block with resource and prompt
   wrappers disabled. Do this before starting a session.
4. **Prove the connection.** `hermes mcp test entroly` must report a successful
   connection and discovered tools. Do not continue silently if the probe fails.
5. **Read or select context.** Use `smart_read` for a known file. Use
   `remember_fragment`, `recall_relevant`, and `optimize_context` when working
   with stored fragments and a token budget. Treat an empty selection as a
   state/setup problem, not as permission to omit all evidence.
6. **Inspect the result.** Call `explain_context` after optimization and surface
   source names, omissions, warnings, and retrieval handles to the user.
7. **Create recovery evidence when required.** For exact omission recovery,
   call `create_context_receipt` with `recoverable=true` and retain the returned
   receipt and local recovery state. `create_context_receipt_from_path` is a
   convenience for `.md`, `.txt`, and `.rst` inputs; it is not the recoverable
   path.
8. **Verify cautiously.** `verify_response` is an additional local signal. Keep
   source inspection and the project's real tests as the authority for coding
   claims.

## Pitfalls

- **MCP is agent-mediated.** It exposes tools to Hermes; it does not
  transparently compress every request or support an automatic all-provider
  interception path.
- **Do not register `entroly serve`.** The installed package's bare `entroly`
  command detects MCP stdio correctly. The explicit `serve` route may enter the
  Docker launcher unless separately configured.
- **Base and proxy installs differ.** `entroly` is sufficient here;
  `entroly[proxy]` is required only for the separate HTTP proxy.
- **A successful install is not a successful connection.** Always run
  `hermes mcp test entroly`, then reload MCP or start a fresh session.
- **Do not broaden the tool surface silently.** Entroly may add tools over time;
  review new tools before adding them to `include`.
- **Receipts can omit content by design.** Exact recovery depends on
  `recoverable=true`, the receipt JSON, and its matching local recovery state.
  Preserve all three until recovery is no longer needed.
- **Local state is project-visible.** Startup and recovery workflows may create
  `.entroly/` in the repository. Inspect it before commits and ignore it when
  appropriate, but do not delete it while recoverable receipts still depend on
  that state.
- **Path-based receipts accept document formats only.** For other content, pass
  explicit documents to `create_context_receipt` or use `smart_read` for a
  source file.
- **Compression results are workload-dependent.** Measure token use and retained
  evidence on the user's repository; do not promise a fixed reduction or
  universal accuracy.

## Verification

For an end-user smoke test:

1. Run `hermes mcp test entroly`; require a successful connection and the
   allowlisted tool names.
2. Run `hermes mcp list` and confirm `entroly` is enabled.
3. Start a fresh chat or run `/reload-mcp`.
4. Ask Hermes to call `mcp_entroly_smart_read` on a harmless repository file
   with a small budget and to report the source and any omissions.
5. For recovery-sensitive work, create a tiny receipt with
   `recoverable=true`, recover one omitted chunk, and compare it byte-for-byte
   with the original text before trusting the workflow.

For repository contributors, run the hermetic skill contract test:

```bash
scripts/run_tests.sh tests/skills/test_entroly_context_compression_skill.py -q
```

On Windows without a POSIX shell, run the same test with:

```powershell
python -m pytest tests/skills/test_entroly_context_compression_skill.py -q
```

If the executable is missing, confirm that the Python scripts directory is on
`PATH` and rerun `python -m pip install -U entroly`. If the MCP test connects but
the tools are absent, inspect the active profile's `tools.include` entries,
restart from the intended repository root, and test again.
