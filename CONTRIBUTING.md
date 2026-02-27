# Contributing to Hermes Agent

Thank you for your interest in contributing to Hermes Agent! This document provides guidelines and information for contributors.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- uv (recommended) or pip for package management
- Git

### Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/hermes-agent.git
   cd hermes-agent
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -e .
   ```

3. Set up your API keys in `~/.hermes/.env`:
   ```bash
   OPENROUTER_API_KEY=your_key_here
   # or other provider keys
   ```

## Making Contributions

### Types of Contributions

We welcome:

- **Bug fixes** - Fix issues reported in GitHub Issues
- **Documentation** - Improve README, docstrings, or add examples
- **Tests** - Add or improve test coverage
- **Features** - Implement features from TODO.md or Issues
- **Tools** - Add new tools or improve existing ones
- **Skills** - Create new skills for the skills registry

### Contribution Workflow

1. **Check existing issues** - Look for open issues before starting work
2. **Create an issue** - For significant changes, open an issue first to discuss
3. **Fork and branch** - Create a feature branch from `main`:
   ```bash
   git checkout -b fix/issue-description
   # or
   git checkout -b feat/feature-description
   ```
4. **Make changes** - Follow the code style guidelines below
5. **Test** - Ensure your changes work and don't break existing functionality
6. **Commit** - Use conventional commit messages (see below)
7. **Push and PR** - Push your branch and open a pull request

### Commit Message Format

We use conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation only
- `style` - Code style (formatting, semicolons, etc.)
- `refactor` - Code refactoring
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

Examples:
```
feat(cli): add /verbose command for detailed output
fix(gateway): resolve hooks initialization order
docs: update installation instructions
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints where practical
- Add docstrings to public functions and classes
- Keep functions focused and reasonably sized
- Prefer readability over cleverness

### Testing

Before submitting:

1. Verify Python syntax:
   ```bash
   python -m py_compile your_file.py
   ```

2. Run existing tests:
   ```bash
   python -m pytest tests/
   ```

3. Test your changes manually in the CLI or gateway

## Project Structure

```
hermes-agent/
├── cli.py              # Main CLI interface
├── run_agent.py        # Agent runtime
├── gateway/            # Messaging platform adapters
│   ├── run.py          # Gateway runner
│   ├── adapters/       # Platform-specific adapters
│   └── hooks.py        # Event hook system
├── tools/              # Tool implementations
├── skills/             # Built-in skills
├── hermes_cli/         # CLI utilities
└── tests/              # Test suite
```

## Adding New Tools

1. Create a new file in `tools/` or add to an existing toolset
2. Define the tool function with proper docstring
3. Create the JSON schema for parameters
4. Register in `tools/__init__.py`
5. Add to appropriate toolset in `toolsets.py`

Example:
```python
def my_tool(param1: str, param2: int = 10) -> str:
    """One-line description.
    
    Detailed description of what the tool does.
    
    Args:
        param1: Description of param1
        param2: Description of param2
        
    Returns:
        Description of return value
    """
    # Implementation
    return result

MY_TOOL_SCHEMA = {
    "name": "my_tool",
    "description": "One-line description.",
    "parameters": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "integer", "description": "...", "default": 10}
        },
        "required": ["param1"]
    }
}
```

## Adding New Skills

Skills live in `~/.hermes/skills/` and are YAML-based:

```yaml
name: my-skill
description: What this skill does
version: "1.0.0"
author: your-name

tools:
  - name: skill_tool
    description: What this tool does
    parameters:
      - name: param1
        type: string
        required: true
```

## Getting Help

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Discord**: Join the Nous Research Discord community

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.
