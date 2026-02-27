# Contributing to Hermes Agent

Thank you for your interest in contributing to Hermes Agent! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.11+
- An OpenRouter API key (for running the agent)
- Git

### Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/NousResearch/hermes-agent.git
   cd hermes-agent
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   # Or using uv
   uv pip install -e .
   ```

3. Copy the example environment file and configure:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. Run the setup script (optional, for shell autocompletion):
   ```bash
   ./setup-hermes.sh
   ```

## Project Structure

```
hermes-agent/
├── run_agent.py          # Main AIAgent class
├── cli.py                # Interactive CLI
├── model_tools.py        # Tool registry orchestration
├── toolsets.py           # Toolset definitions
├── agent/                # Agent internals (extracted modules)
│   ├── prompt_builder.py   # System prompt assembly
│   ├── context_compressor.py
│   ├── auxiliary_client.py
│   └── ...
├── tools/                # Individual tool implementations
│   ├── registry.py         # Central tool registry
│   ├── terminal_tool.py
│   ├── web_tools.py
│   ├── file_tools.py
│   └── ...
├── gateway/              # Multi-platform messaging gateway
│   ├── run.py
│   ├── platforms/          # Platform adapters (Telegram, Discord, etc.)
│   └── ...
├── skills/               # Built-in skills
├── docs/                 # Documentation
└── tests/                # Test suite
```

## Contributing Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Add docstrings to functions and classes (Google-style docstrings preferred)
- Keep lines under 100 characters when reasonable

### Adding a New Tool

Tools self-register with the central registry. To add a new tool:

1. Create a new file in `tools/` (e.g., `tools/my_tool.py`)

2. Define your tool handler and schema:
   ```python
   #!/usr/bin/env python3
   """
   My Tool Module - Brief description
   
   Longer description of what the tool does.
   """
   
   import json
   from tools.registry import registry
   
   
   def my_tool_handler(args: dict, **kwargs) -> str:
       """Execute the tool and return JSON result."""
       # Your implementation here
       return json.dumps({"result": "success"})
   
   
   def check_my_tool_requirements() -> bool:
       """Check if tool dependencies are available."""
       return True  # Or actual availability check
   
   
   MY_TOOL_SCHEMA = {
       "name": "my_tool",
       "description": "What this tool does...",
       "parameters": {
           "type": "object",
           "properties": {
               "param1": {
                   "type": "string",
                   "description": "Description of param1"
               }
           },
           "required": ["param1"]
       }
   }
   
   # Register with the central registry
   registry.register(
       name="my_tool",
       toolset="my_toolset",
       schema=MY_TOOL_SCHEMA,
       handler=lambda args, **kw: my_tool_handler(args, **kw),
       check_fn=check_my_tool_requirements,
   )
   ```

3. Add the import to `model_tools.py` in `_discover_tools()`:
   ```python
   _modules = [
       # ... existing modules ...
       "tools.my_tool",
   ]
   ```

4. Add your toolset to `toolsets.py` if it's a new category

### Adding a Skill

Skills are markdown documents with YAML frontmatter. Create a new skill:

1. Create a directory in `skills/`:
   ```
   skills/my-skill/
   └── SKILL.md
   ```

2. Write the skill file with proper frontmatter:
   ```markdown
   ---
   name: my-skill
   description: Brief description of what this skill does
   version: 1.0.0
   author: Your Name
   tags: [category, subcategory]
   ---
   
   # My Skill
   
   Instructions for the agent when using this skill...
   ```

### Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b feat/my-feature
   # or
   git checkout -b fix/issue-description
   ```

2. **Make your changes** with clear, focused commits

3. **Test your changes**:
   ```bash
   # Run the test suite
   pytest tests/
   
   # Test manually with the CLI
   python cli.py
   ```

4. **Update documentation** if needed

5. **Submit a pull request** with:
   - Clear title following conventional commits (e.g., `feat(tools):`, `fix(cli):`, `docs:`)
   - Description of what changed and why
   - Reference to any related issues

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding or correcting tests
- `chore`: Changes to build process or auxiliary tools

Scopes: `cli`, `gateway`, `tools`, `skills`, `agent`, etc.

### Security Considerations

When contributing tools that interact with external resources:

- **Skills Guard**: External skills pass through security scanning (`tools/skills_guard.py`)
- **Dangerous Commands**: Terminal commands are checked against patterns (`tools/approval.py`)
- **Memory Scanning**: Memory entries are scanned for injection attempts
- **Context Scanning**: AGENTS.md and similar files are scanned before prompt injection

If your change affects security, please note this in your PR.

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include steps to reproduce for bugs
- Include system information (OS, Python version)
- Check existing issues before creating duplicates

## Questions?

- Open a GitHub Discussion for general questions
- Join the Nous Research community for real-time chat

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
