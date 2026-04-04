# A2A Protocol — User Guide

This guide explains how to use the A2A (Agent-to-Agent) protocol support in Hermes Agent — both as a **server** (making your Hermes agent discoverable) and as a **client** (calling other A2A agents from within a conversation).

---

## What is A2A?

A2A (Agent-to-Agent) is an open standard under the Linux Foundation for agent discovery and communication. It answers the question *"who can help me?"* — complementing MCP which answers *"what tools can I use?"*

Any A2A-compatible system — Google Vertex AI Agent Engine, LangGraph, CrewAI, AutoGen, or another Hermes instance — can discover and call your agent without knowing anything about Hermes internals.

---

## Part 1 — Running Hermes as an A2A Server

### Installation

```bash
pip install 'hermes-agent[a2a]'
```

This installs `a2a-sdk` and `uvicorn` alongside your existing Hermes setup.

### Starting the A2A server

```bash
hermes-a2a
```

Or equivalently:

```bash
python -m a2a_adapter
```

By default the server starts on `http://0.0.0.0:9000`.

### Configuration

All configuration is via environment variables — no config file changes needed.

| Variable | Default | Description |
|---|---|---|
| `A2A_HOST` | `0.0.0.0` | Bind host |
| `A2A_PORT` | `9000` | Bind port |
| `A2A_KEY` | *(none)* | Optional Bearer token for auth |
| `AGENT_NAME` | `hermes-agent` | Name shown in Agent Card |
| `AGENT_DESCRIPTION` | auto | Description shown in Agent Card |
| `AGENT_SKILLS` | *(none)* | Comma-separated skill names |
| `AGENT_MODEL` | *(none)* | Model name shown in Agent Card metadata |

Example:

```bash
AGENT_NAME=ronny \
AGENT_MODEL="qwen2.5:35b" \
AGENT_SKILLS="coding,analysis,reasoning" \
AGENT_DESCRIPTION="Ronny — code and analysis specialist" \
A2A_PORT=9000 \
hermes-a2a
```

### Verifying the server

Once running, check the Agent Card:

```bash
curl http://localhost:9000/.well-known/agent.json
```

Expected response:

```json
{
  "name": "ronny",
  "description": "Ronny — code and analysis specialist",
  "url": "http://localhost:9000",
  "version": "1.0.0",
  "capabilities": { "streaming": true },
  "skills": [
    { "id": "coding", "name": "coding" },
    { "id": "analysis", "name": "analysis" },
    { "id": "reasoning", "name": "reasoning" }
  ],
  "metadata": { "model": "qwen2.5:35b" }
}
```

Send a test task:

```bash
curl -X POST http://localhost:9000 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "test-1",
    "method": "tasks/send",
    "params": {
      "id": "task-1",
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "Hello, what can you do?"}]
      }
    }
  }'
```

### Running alongside the gateway

The A2A server (`hermes-a2a`) runs as a **separate process** on a separate port from the Hermes gateway. Run both together:

```bash
# Terminal 1 — Hermes gateway (OpenAI-compatible)
hermes gateway

# Terminal 2 — A2A server
AGENT_NAME=myagent A2A_PORT=9000 hermes-a2a
```

In Docker, expose both ports:

```bash
docker run -d \
  -p 8642:8642 \   # Hermes gateway
  -p 9000:9000 \   # A2A server
  -e AGENT_NAME=ronny \
  -e AGENT_MODEL="qwen2.5:35b" \
  -e AGENT_SKILLS="coding,analysis" \
  myagent-image
```

### Registering with an orchestrator (e.g. Akela Pack)

1. In the Pack UI, add a new agent
2. Set the endpoint to `http://your-server-ip:9000`
3. Set protocol to **A2A**
4. Click **Discover** — the UI auto-fills name, skills, and model from the Agent Card

---

## Part 2 — Calling Remote A2A Agents from Hermes

Once the `a2a` toolset is enabled, Hermes can call any A2A-compatible agent as part of a conversation.

### Enabling the A2A tools

Add `a2a` to your toolset in `~/.hermes/config.yaml`:

```yaml
enabled_toolsets:
  - hermes-core
  - a2a
```

Or enable just for specific sessions using the `--toolsets` flag:

```bash
hermes chat --toolsets hermes-core,a2a
```

### Configuring named agents

Add remote agents to `~/.hermes/config.yaml` so Hermes knows about them by name:

```yaml
a2a_agents:
  researcher:
    url: http://192.168.1.100:9000
    description: "Deep research and web analysis agent"
  coder:
    url: http://192.168.1.101:9000
    description: "Code writing and debugging agent"
    bearer_token: "sk-abc123"   # if the agent requires auth
  legal:
    url: https://legal-agent.company.internal:9000
    description: "Legal document review agent"
```

### Available tools

#### `a2a_discover` — learn what a remote agent can do

Fetches the Agent Card and returns the agent's name, description, skills, and model.

Example prompt:
> *"Discover the agent at http://192.168.1.100:9000 and tell me what it can do."*

Hermes will call `a2a_discover` and report back the agent's capabilities.

#### `a2a_call` — send a task to a remote agent

Calls the remote agent with a message and returns its response. Automatically uses SSE streaming (`tasks/sendSubscribe`) when the agent's Agent Card advertises `capabilities.streaming: true`, falling back to `tasks/send` otherwise.

Example prompts:
> *"Ask the researcher agent to find the latest papers on quantum error correction."*

> *"Use the coder agent to write a Python function that parses JWT tokens."*

> *"Call http://192.168.1.102:9000 and ask it to summarize this document: [...]"*

For multi-turn conversations with the same remote agent, reuse a session ID:

> *"Start a conversation with the coder agent about refactoring our auth module. Keep using the same session so it remembers context."*

Hermes will automatically maintain a `session_id` across calls.

To override streaming detection pass `stream=true` or `stream=false` explicitly.

#### `a2a_local_scan` — find running A2A agents on localhost

Scans a range of localhost ports for running A2A agents without requiring any pre-configuration. Useful when you don't know which ports agents are running on.

Example prompts:
> *"Find all local agents running on this machine."*

> *"Scan ports 9000 to 9005 for A2A agents."*

Hermes will probe each port for a `/.well-known/agent.json` endpoint and return a list of discovered agents with their names, descriptions, and skills. You can then call any discovered agent directly.

Default scan range is ports `9000–9010`. All three parameters are optional:

| Parameter | Default | Description |
|---|---|---|
| `host` | `localhost` | Host to scan |
| `port_start` | `9000` | First port to probe |
| `port_end` | `9010` | Last port to probe (max range: 100 ports) |

### Multi-agent workflow example

Here is a complete example using two remote agents:

> *"I need to research quantum computing applications in cryptography. Use the researcher agent to gather information, then have the coder agent write a Python demo based on the findings."*

Hermes will:
1. Call `a2a_call` on the researcher agent with the research task
2. Receive the research summary
3. Call `a2a_call` on the coder agent, passing the research as context
4. Return the combined result

---

## Supported A2A Methods

| JSON-RPC Method | Support | Description |
|---|---|---|
| `tasks/send` | ✅ Full | Non-streaming task — waits for completion |
| `tasks/sendSubscribe` | ✅ Full | Streaming SSE — incremental token updates |
| `tasks/cancel` | ✅ | Cancel an in-progress task |
| `tasks/get` | SDK-managed | Handled internally by task store |

---

## Security

### Protecting your A2A server

Set `A2A_KEY` to require Bearer token authentication:

```bash
A2A_KEY=my-secret-token hermes-a2a
```

All requests must then include:
```
Authorization: Bearer my-secret-token
```

### Calling authenticated agents

Set `bearer_token` in the agent config:

```yaml
a2a_agents:
  secure_agent:
    url: http://192.168.1.100:9000
    bearer_token: "my-secret-token"
```

Or pass it directly in your prompt:
> *"Call the agent at http://192.168.1.100:9000 with bearer token 'abc123' and ask it to..."*

---

## Troubleshooting

**Agent Card returns 404**
- The A2A server is not running, or the port is wrong
- Run `curl http://host:port/.well-known/agent.json` to verify

**tasks/send returns 401**
- The agent requires a Bearer token — set `A2A_KEY` or `bearer_token` in config

**tasks/send returns an LLM error**
- The underlying Hermes model is not configured. Run `hermes setup` inside the agent container to configure the model provider.

**"a2a-sdk is not installed" error**
- Run `pip install 'hermes-agent[a2a]'`

**Tool `a2a_call` not available in chat**
- Add `a2a` to `enabled_toolsets` in `~/.hermes/config.yaml`
