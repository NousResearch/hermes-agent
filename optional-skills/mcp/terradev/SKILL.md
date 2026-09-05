---
name: terradev
description: "Provision GPU instances across 17 cloud providers."
version: 1.0.0
author: Theo Wolfenden
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [GPU, compute, cloud, MCP, provisioning]
    category: mcp
    related_skills: []
prerequisites:
  commands: [terradev]
  mcp_servers: [terradev]
---

# Terradev GPU Provisioning

Use the `terradev` MCP server to provision, inspect, and terminate GPU instances
across 17 cloud providers without leaving the agent.

## Prerequisites

- Install: `pip install terradev-cli`
- Configure at least one provider: `terradev configure --provider runpod`
- Register the MCP server in your Hermes config:

```yaml
mcpServers:
  terradev:
    transport: stdio
    command: terradev
    args: [mcp, serve]
```

## When to Use

- User asks to provision a GPU for training, inference, or batch work
- Checking which GPU providers have capacity right now
- Terminating an instance after a job completes

## When NOT to Use

- Kubernetes cluster management → use the `kubectl` skill
- Serverless inference → use provider-native tools directly
- Jobs that run in under 10 minutes (cold-start overhead may exceed runtime)

## Key MCP Tools

| Tool | What it does |
|---|---|
| `list_providers` | Current availability and spot prices across all providers |
| `provision_instance` | Provision cheapest matching GPU; returns `instance_id` and `address` |
| `get_instance_status` | Poll until IP is assigned |
| `terminate_instance` | Terminate by `instance_id` |

## Example Prompts

> "Find the cheapest H100 available right now."

> "Provision an A100 for under $2/hr and give me the SSH address."

> "Terminate instance `runpod-abc123`."

## Supported Providers

RunPod, Vast.ai, TensorDock, Crusoe, Hyperstack, Latitude, E2E Networks,
Gcore, AWS, GCP, Azure, DigitalOcean, InferX, Baseten, SiliconFlow,
HuggingFace, YottaLabs.
