# Tool Call Benchmark: Gemma 4 vs mimo-v2-pro

Date: 2026-04-13
Status: Awaiting execution

## Test Design

100 diverse tool calls across 7 categories:

| Category | Count | Tools Tested |
|----------|-------|--------------|
| File operations | 20 | read_file, write_file, search_files |
| Terminal commands | 20 | terminal |
| Web search | 15 | web_search |
| Code execution | 15 | execute_code |
| Browser automation | 10 | browser_navigate |
| Delegation | 10 | delegate_task |
| MCP tools | 10 | mcp_* |

## Metrics

| Metric | mimo-v2-pro | Gemma 4 |
|--------|-------------|---------|
| Schema parse success | — | — |
| Tool execution success | — | — |
| Parallel tool success | — | — |
| Avg latency (s) | — | — |
| Token cost per call | — | — |

## How to Run

```bash
# Single model
python3 benchmarks/tool_call_benchmark.py --model nous:xiaomi/mimo-v2-pro
python3 benchmarks/tool_call_benchmark.py --model ollama/gemma4:latest

# Both + comparison
python3 benchmarks/tool_call_benchmark.py --compare
```

## Results

Awaiting execution. Run the benchmark to populate.

## Gemma 4-Specific Failure Modes

To be documented after benchmark execution. Look for:
- JSON schema violations (malformed arguments)
- Wrong tool selection (correct schema, wrong tool)
- Hallucinated tool names (tool doesn't exist)
- Missing required arguments
- Extra/phantom arguments
- Parallel call failures (multiple tool_calls where one was expected)
