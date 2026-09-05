# Terradev Integration

Adds GPU provisioning capability to Hermes agents via Terradev's MCP server.
Agents can autonomously route compute requests across 17 GPU providers without
human intervention. No new dependencies — Terradev registers as an MCP server
and Hermes calls it natively.

## Prerequisites

```bash
pip install terradev-cli
terradev configure --provider runpod   # or any of the 17 providers
```

## MCP server config

Add to your Hermes MCP configuration:

```yaml
mcpServers:
  terradev:
    transport: stdio
    command: terradev
    args: [mcp, serve]
    env:
      TERRADEV_RUNPOD_API_KEY: "${RUNPOD_API_KEY}"
```

This registers Terradev's full MCP surface (261 tools) with the agent runtime.

## Using the skill

The `skills/terradev.skill.yaml` file in this repo registers three focused
tools for GPU lifecycle management:

| Tool | What it does |
|---|---|
| `list_gpu_providers` | Returns current availability and spot prices across all providers |
| `provision_gpu` | Picks the cheapest matching instance and provisions it |
| `terminate_gpu` | Terminates an instance by ID |

## End-to-end example

1. **Configure the MCP server** (see above)
2. **Ask Hermes to provision a GPU:**

   > "Provision an H100 for under $3/hr and run my training script"

   Hermes calls `list_gpu_providers`, finds RunPod at $2.49/hr, calls
   `provision_gpu`, gets back an instance ID and IP address.

3. **Watch it route across providers:**

   If RunPod is unavailable, Terradev automatically tries the next cheapest
   provider (Vast.ai, TensorDock, Crusoe, etc.) with no agent-side changes.

4. **Cleanup is automatic:**

   The agent calls `terminate_gpu` when done. If it forgets, Terradev's
   `--max-runtime` flag hard-terminates the instance after a TTL.

## Supported providers

RunPod, Vast.ai, TensorDock, Crusoe, Hyperstack, Latitude, E2E Networks,
Gcore, AWS, GCP, Azure, DigitalOcean, InferX, Baseten, SiliconFlow,
HuggingFace, YottaLabs.

## Links

- [Terradev on GitHub](https://github.com/theoddden/Terradev)
- [Terradev MCP reference](https://github.com/theoddden/Terradev/blob/main/docs/mcp.md)
