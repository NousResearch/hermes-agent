# Technical Context: Hermes-Agent

## Technologies Used

### Core Stack
- **Python 3.11+** - Primary language
- **OpenAI SDK** - For LLM API interactions (OpenAI-compatible)
- **OpenRouter** - Default LLM provider (supports multiple models)
- **Rich** - Terminal formatting and panels
- **prompt_toolkit** - Interactive input with history
- **Fire** - CLI argument parsing
- **PyYAML** - Configuration files
- **python-dotenv** - Environment variable management

### Tool Dependencies
- **Firecrawl** - Web search and extraction (`FIRECRAWL_API_KEY`)
- **mini-swe-agent** - Terminal tool backend (local/docker/singularity/modal/ssh)
- **agent-browser** - Browser automation (npm package)
- **Browserbase** - Cloud browser execution (`BROWSERBASE_API_KEY`)
- **FAL.ai** - Image generation with FLUX (`FAL_KEY`)
- **Nous API** - Vision and MoA tools (`NOUS_API_KEY`)

### Optional Dependencies
- **Modal** - Cloud compute for sandboxed environments
- **Singularity/Apptainer** - Rootless containers (HPC environments)
- **Docker** - Container isolation

## Development Setup

### Quick Start
```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/NousResearch/Hermes-Agent.git
cd Hermes-Agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e ./mini-swe-agent

# Install browser tools (optional)
npm install

# Configure environment
cp .env.example .env
# Edit .env with your API keys
```

### Key Configuration Files
- `.env` - API keys and secrets
- `cli-config.yaml` - CLI configuration (model, terminal, toolsets, personalities)
- `configs/` - Batch run scripts and configuration

### Environment Variables

**Required for Full Functionality:**
- `OPENROUTER_API_KEY` - Primary LLM access
- `FIRECRAWL_API_KEY` - Web tools
- `NOUS_API_KEY` - Vision and reasoning tools
- `FAL_KEY` - Image generation

**Terminal Backend:**
- `TERMINAL_ENV` - Backend type: `local`, `docker`, `singularity`, `modal`, `ssh`
- `TERMINAL_CWD` - Working directory
- `TERMINAL_DOCKER_IMAGE` / `TERMINAL_SINGULARITY_IMAGE` - Container images
- `TERMINAL_SSH_HOST/USER/KEY` - SSH backend config
- `SUDO_PASSWORD` - Optional sudo support

**Browser:**
- `BROWSERBASE_API_KEY` - Browser automation
- `BROWSERBASE_PROJECT_ID` - Browserbase project

## Technical Constraints

1. **Context Window Limits** - Long tool outputs can exhaust context; trajectory compression helps
2. **API Rate Limits** - OpenRouter and tool APIs have rate limits; exponential backoff implemented
3. **Tool Availability** - Tools gracefully degrade if dependencies/keys missing
4. **Async Compatibility** - Some tools are async, handled via `asyncio.run()` in sync context

## Dependency Graph

```
tools/*.py → tools/__init__.py → model_tools.py → toolsets.py → toolset_distributions.py
                                       ↑
run_agent.py ──────────────────────────┘
cli.py → run_agent.py (uses AIAgent with quiet_mode=True)
batch_runner.py → run_agent.py + toolset_distributions.py
```

## Tool Usage Patterns

### Adding a New Tool
1. Create `tools/your_tool.py` with handler + requirements check
2. Export in `tools/__init__.py`
3. Register in `model_tools.py` (definitions + handler routing)
4. Add to toolset in `toolsets.py`
5. Optionally add to `toolset_distributions.py` for batch processing

### Tool Handler Pattern
```python
def your_tool(param: str, task_id: str = None) -> str:
    """Execute tool and return JSON string result."""
    try:
        result = {"success": True, "data": "..."}
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)
```

All tool handlers MUST return a JSON string, never raw dicts.
