### Install and Quick Start Hermes Agent

Source: https://github.com/nousresearch/hermes-agent/blob/main/skills/autonomous-ai-agents/hermes-agent/SKILL.md

Commands for installing the agent via shell script and performing common initial actions like starting a chat or running the setup wizard.

```bash
# Install
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash

# Interactive chat (default)
hermes

# Single query
hermes chat -q "What is the capital of France?"

# Setup wizard
hermes setup

# Change model/provider
hermes model

# Check health
hermes doctor
```

--------------------------------

### Installation and Quick Start

Source: https://github.com/nousresearch/hermes-agent/blob/main/skills/autonomous-ai-agents/hermes-agent/SKILL.md

Installation script and basic commands to get Hermes Agent running. Includes interactive chat, single query execution, setup wizard, model configuration, and health checks.

```APIDOC
## Installation

### Description
Install Hermes Agent using the official installation script.

### Command
```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

---

## Interactive Chat

### Description
Start Hermes Agent in interactive chat mode (default behavior).

### Command
```bash
hermes
```

---

## Single Query

### Description
Execute a single query without entering interactive mode.

### Command
```bash
hermes chat -q "What is the capital of France?"
```

### Parameters
- **-q, --query** (string) - Required - The query text to execute

---

## Setup Wizard

### Description
Run the interactive setup wizard to configure Hermes Agent.

### Command
```bash
hermes setup
```

---

## Change Model/Provider

### Description
Interactively select and configure the LLM model and provider.

### Command
```bash
hermes model
```

---

## Health Check

### Description
Verify Hermes Agent installation and configuration status.

### Command
```bash
hermes doctor
```
```

--------------------------------

### Install Hermes Agent

Source: https://github.com/nousresearch/hermes-agent/blob/main/README.md

This command downloads and executes the Hermes Agent installation script. It works on Linux, macOS, WSL2, and Android via Termux.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

--------------------------------

### Install Hermes Agent via one-line script

Source: https://github.com/nousresearch/hermes-agent/blob/main/website/docs/getting-started/installation.md

The installer automatically handles dependencies like Python, Node.js, and ripgrep for Linux, macOS, WSL2, and Android.

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

--------------------------------

### Quick Start Development Setup for Hermes Agent (Bash)

Source: https://github.com/nousresearch/hermes-agent/blob/main/README.md

Clone the Hermes Agent repository and use the `setup-hermes.sh` script for a quick development environment setup, including `uv` installation, virtual environment creation, and package installation.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent
./setup-hermes.sh     # installs uv, creates venv, installs .[all], symlinks ~/.local/bin/hermes
./hermes              # auto-detects the venv, no need to `source` first
```