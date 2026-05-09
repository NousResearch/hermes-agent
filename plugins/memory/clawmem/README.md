# ClawMem Memory Provider

Shared memory between Jane (OpenClaw) and Hermes agent via ClawMem API.

## Setup

```bash
# Set environment variables
export CLAWMEM_API_TOKEN="877f341cde532adcef45bb6c7a0525b99a95da12"
export CLAWMEM_REPO="main-787c63/memory"

# Or create config file
echo '{
    "api_token": "877f341cde532adcef45bb6c7a0525b99a95da12",
    "repo": "main-787c63/memory",
    "user_id": "hermes"
}' > ~/.hermes/clawmem.json
```

## Tools

| Tool | Description |
|------|-------------|
| `memory_recall` | Search shared memory by meaning |
| `memory_list` | List all memories (filter by kind/topic) |
| `memory_store` | Store a durable fact |
| `memory_forget` | Mark a memory as stale |

## Example

```
memory_recall(query="Jazper's trading style")
memory_store(detail="Jazper uses 100x leverage on crypto", title="Jazper trading", kind="preference", topics=["jazper", "trading"])
```
