---
name: uv-toolchain
description: "Run Python scripts, manage venvs, and dependencies with uv."
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [uv, python, packaging, dependencies, venv, scripts]
    category: software-development
    related_skills: [test-driven-development, systematic-debugging]
---

# `uv` Python Toolchain

Manage Python dependencies, virtual environments, standalone scripts, and project lifecycles with `uv`.

## Core Philosophy & Fast Decision Matrix

`uv` is an extremely fast, standalone Python package and project manager. Use it to avoid slow package installs, global environment pollution, and PEP 668 externally managed environment conflicts.

| Scenario | Recommended Approach | Example Command |
|---|---|---|
| Run a one-off script with packages | `uv run --with <pkg>` | `uv run --with httpx,rich script.py` |
| Self-contained reusable script | PEP 723 Inline Metadata | `uv run standalone_script.py` |
| Run a CLI tool without installing | `uvx <tool>` | `uvx ruff check .` |
| Create and populate a virtualenv | `uv venv` + `uv pip install` | `uv venv && uv pip install -r reqs.txt` |
| Lock dependencies reproducibly | `uv pip compile` | `uv pip compile reqs.in -o reqs.txt` |
| Manage a complete project | `uv init` / `uv add` / `uv sync` | `uv sync --locked` |

---

## Pattern 1: Ephemeral Script Execution & PEP 723

### 1. On-Demand Dependency Execution
When running a quick Python script or inline one-liner that needs third-party packages, do NOT modify the active environment:
```bash
# Run one-liner with temporary packages
uv run --with httpx,rich python -c "import httpx; print(httpx.get('https://httpbin.org/json').json())"

# Run an existing script file with extra dependencies
uv run --with pandas,openpyxl process_data.py
```

### 2. Self-Contained Scripts with PEP 723 Inline Metadata
For standalone automation scripts, declare dependencies directly at the top of the file using standard PEP 723 comments:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "httpx>=0.27.0",
#     "rich>=13.7.0",
#     "pydantic>=2.6.0",
# ]
# ///

import httpx
from rich import print
from pydantic import BaseModel

class User(BaseModel):
    name: str

print(User(name="Hermes"))
```

**Execution**:
```bash
uv run standalone_script.py
```
`uv` automatically creates an isolated, cached environment satisfying the declared dependencies on first run and executes the script seamlessly.

---

## Pattern 2: Run Developer Tools with `uvx`

Run Python CLI tools, formatters, and linters in isolated ephemeral environments without installing them globally or into the project venv:

```bash
# Lint and format code with Ruff
uvx ruff check .
uvx ruff format .

# Run static type checking with MyPy
uvx mypy src/

# Run security scanning with Bandit
uvx bandit -r src/

# Launch temporary local documentation servers
uvx mkdocs serve
```

---

## Pattern 3: Virtual Environments & Pip Compatibility

### 1. Create Virtual Environments
```bash
# Create a standard .venv
uv venv

# Create a venv at a specific path with a specific Python version
uv venv custom-env --python 3.12
```

### 2. Fast Package Installation (`uv pip`)
```bash
# Install packages into the active or discovered .venv
uv pip install fastapi uvicorn

# Install from requirements file
uv pip install -r requirements.txt

# Install packages into a specific virtualenv path
uv pip install --python custom-env/ -r requirements.txt
```

### 3. Compile Deterministic Lockfiles
```bash
# Compile loose dependencies into pinned, hashed requirements
uv pip compile requirements.in -o requirements.txt

# Upgrade pinned versions
uv pip compile --upgrade requirements.in -o requirements.txt
```

---

## Pattern 4: Project Workspaces & Full Lifecycle

For Python applications or libraries managed with `pyproject.toml`:

```bash
# Initialize a new project
uv init my-project
cd my-project

# Add runtime dependencies
uv add requests pydantic

# Add development/testing dependencies
uv add --dev pytest ruff

# Install and synchronize all dependencies exactly to uv.lock
uv sync

# Run project tests or scripts inside the project environment
uv run pytest
```

---

## Pattern 5: Multi-Python Version Management

`uv` can download, install, and manage official Python interpreter binaries independently of system package managers:

```bash
# List available and installed Python versions
uv python list

# Install specific Python versions
uv python install 3.11 3.12 3.13

# Pin Python version for the current directory (.python-version)
uv python pin 3.12

# Run a script explicitly with a specific Python interpreter
uv run --python 3.13 script.py
```

---

## Troubleshooting & Best Practices

1. **Clean Cache**: If a package build gets corrupted or fails mid-stream:
   ```bash
   uv cache clean
   ```
2. **Offline & Air-Gapped Environments**:
   ```bash
   uv run --offline script.py
   ```
3. **Avoid Manual Activation**: Prefer `uv run <cmd>` instead of `source .venv/bin/activate` or `.\.venv\Scripts\Activate.ps1`. `uv run` handles environment activation, path resolution, and lockfile synchronization automatically in a single step.
