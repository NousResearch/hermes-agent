# cosmos API Reference

## Core Concepts

cosmos is designed around three main pillars:
1. **Memory**: Persistent storage for context, facts, and relationships.
2. **Agents**: Specialist workers for code, reasoning, and research.
3. **Evolution**: Self-improving feedback loops.

## Python SDK

The `cosmosClient` is the primary entry point for Python integrations.

```python
from cosmos.client import cosmosClient

client = cosmosClient()
client.remember("Project X deadline is Friday")
```

### Methods

#### `remember(content: str, tags: list[str] = None)`
Stores information in the semantic and episodic memory layers.

#### `recall(query: str, limit: int = 5)`
Retrieves relevant memories based on semantic similarity.

#### `delegate_task(task: str)`
Dispatches a task to the Agent Swarm. The system automatically selects the best agent (Planner, Coder, etc.).

## MCP Protocol

For integration with Claude Code or other MCP-compatible IDEs, cosmos exposes the following tools and resources.

### Tools

#### `cosmos_remember`
Store information in long-term memory.

**Parameters:**
- `content` (string, required): The information to remember
- `tags` (array[string], optional): Tags for categorization
- `importance` (number, optional): Importance score 0-1 (default: 0.5)

#### `cosmos_recall`
Search and retrieve relevant memories.

**Parameters:**
- `query` (string, required): Search query
- `limit` (integer, optional): Maximum results (default: 5)

#### `cosmos_delegate`
Delegate a task to a specialist agent.

**Parameters:**
- `task` (string, required): The task to delegate
- `agent_type` (string, optional): Type of specialist: 'code', 'reasoning', 'research', 'creative', or 'auto' (default: auto)

#### `cosmos_evolve`
Provide feedback for system improvement.

**Parameters:**
- `feedback` (string, required): Your feedback on the system's performance

#### `cosmos_status`
Get the current system status including memory statistics, active agents, and evolution metrics.

#### `cosmos_vision`
Analyze an image using the vision module.

**Parameters:**
- `image` (string, required): Image path, URL, or base64 string
- `task` (string, optional): Task type: 'caption', 'vqa', 'ocr', 'classify' (default: caption)

#### `cosmos_browse`
Use the intelligent web agent to browse the internet.

**Parameters:**
- `goal` (string, required): What to accomplish or find
- `url` (string, optional): Starting URL

#### `cosmos_export`
Export conversation history, memories, and context to a shareable format.

**Parameters:**
- `format` (string, optional): Export format - 'json', 'markdown'/'md', 'html', or 'text'/'txt' (default: markdown)
- `include_memories` (boolean, optional): Include stored memories (default: true)
- `include_conversations` (boolean, optional): Include conversation history (default: true)
- `include_knowledge_graph` (boolean, optional): Include entities and relationships (default: true)
- `include_statistics` (boolean, optional): Include memory statistics (default: true)
- `start_date` (string, optional): Only include items after this date (ISO format: YYYY-MM-DD)
- `end_date` (string, optional): Only include items before this date (ISO format: YYYY-MM-DD)
- `tags` (array[string], optional): Only include items with these tags
- `output_path` (string, optional): Custom output file path

**Example:**
```json
{
  "format": "html",
  "include_memories": true,
  "include_conversations": true,
  "tags": ["project", "important"],
  "start_date": "2025-01-01"
}
```

#### `cosmos_list_exports`
List all available conversation exports.

### Resources

- `cosmos://memory/recent`: View active context and recent memories
- `cosmos://memory/graph`: Knowledge graph entities and relationships
- `cosmos://agents/active`: Currently running specialist agents
- `cosmos://evolution/fitness`: System performance and evolution metrics
- `cosmos://proactive/suggestions`: Anticipatory suggestions from the proactive agent
- `cosmos://system/health`: Real-time health status and metrics
- `cosmos://exports/list`: List of all available conversation exports

## Best Practices

1. **Tagging**: Use consistent tags when storing memories to improve recall accuracy.
2. **Feedback**: Regularly use `delegate_task` with feedback enabled to help the system evolve.
3. **Context**: Keep active context windows clean by archiving old tasks.
