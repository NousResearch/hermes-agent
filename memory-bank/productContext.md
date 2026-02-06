# Product Context: Hermes-Agent

## Why This Project Exists

Hermes-Agent addresses several key challenges in the AI agent space:

1. **Unified Tool Interface** - Provides a clean, consistent interface for LLMs to use various tools (web, terminal, browser, vision, etc.) without requiring custom integration for each model provider.

2. **Training Data Generation** - Enables efficient generation of high-quality tool-calling trajectories for fine-tuning LLMs, with features like batch processing, checkpointing, and trajectory compression.

3. **Flexible Deployment** - Supports multiple execution environments (local, Docker, Singularity, Modal, SSH) to accommodate different security and isolation requirements.

4. **Developer Experience** - Offers a beautiful, interactive CLI with kawaii-style feedback that makes working with AI agents enjoyable.

## Problems It Solves

### For AI Researchers
- **Data Generation at Scale**: Parallel batch processing with content-based checkpointing for fault tolerance
- **Clean Trajectories**: Trajectory compression to fit token budgets while preserving important information
- **Toolset Distributions**: Probability-based tool selection for varied training data

### For Developers
- **Tool Orchestration**: Logical grouping of tools into toolsets (research, development, debugging, etc.)
- **Session Persistence**: Conversation history and session logging for debugging
- **Multi-Model Support**: Works with any OpenAI-compatible API (OpenRouter, local models, etc.)

### For MLOps
- **Skills System**: On-demand knowledge documents for specific tools/frameworks (Axolotl, vLLM, TRL, etc.)
- **Sandboxed Execution**: Terminal commands can run in isolated environments (Docker, Singularity, Modal)
- **Configurable Backends**: Easy switching between local and cloud execution

## How It Should Work

### User Flow (CLI)
1. User launches `./hermes` 
2. Beautiful welcome banner displays with caduceus logo, model info, and available tools
3. User types a natural language request
4. Agent processes request, potentially calling tools with animated feedback
5. Agent responds with results, conversation continues
6. Session is automatically logged for debugging

### User Flow (Batch Processing)
1. User prepares JSONL file with prompts
2. Runs `batch_runner.py` with distribution and worker count
3. System processes prompts in parallel, saves checkpoints
4. Completed trajectories saved to `data/<run_name>/trajectories.jsonl`
5. Optional: compress trajectories with `trajectory_compressor.py`

## User Experience Goals

- **Delightful Interaction**: Kawaii ASCII faces, animated spinners, cute messages
- **Informative Feedback**: Clear progress indication during tool execution
- **Configurable Personalities**: From "helpful" to "pirate" to "Shakespeare"
- **Easy Configuration**: YAML config file + environment variables + CLI flags
- **Graceful Degradation**: Missing tools/APIs don't break the system, just disable features
