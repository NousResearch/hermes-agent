# cosmos Setup Guide 🛠️

Welcome to cosmos! This guide provides detailed instructions on configuring the environment, API keys, and local services.

> **🔒 Privacy First**: cosmos is designed to run 100% locally. No data is sent to the cloud unless you explicitly enable an external integration (like X.com or Grok). All memories, vector databases, and keyrings are stored in your `./data` folder.

---

## 📋 Feature Prerequisites

Before running the Setup Wizard, ensure you have the necessary prerequisites for the features you intend to use.

| Feature Suite | Requires | Details |
|---------------|----------|---------|
| **Core Cognition** | Python 3.10+ | Runs on CPU/local RAM. No keys needed. |
| **Local Inference/RAG** | [Ollama](https://ollama.ai) | Suggested models: `mistral`, `nomic-embed-text`. |
| **DeGen Mob (Solana)** | `HELIUS_API_KEY` | Get free key at [dev.helius.xyz](https://dev.helius.xyz). |
| **Elite Trading** | `SOLANA_PRIVATE_KEY` | **Safety Warning**: Use a burner wallet with minimal funds. |
| **Grok X Search** | `XAI_API_KEY` | Get key at [x.ai](https://x.ai) for live Twitter access. |
| **Discord Bridge** | `DISCORD_TOKEN` | Create a bot on [Discord Developer Portal](https://discord.com/developers). |
| **GitHub Integration** | `GITHUB_TOKEN` | Personal Access Token (Classic) with repo scopes. |

---

## 🔧 Step-by-Step Installation

### 1. Environment Setup

```bash
# Verify Python
python --version  # Should be 3.10 or higher

# Create Virtual Environment (Recommended)
python -m venv venv
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU Support**: If you have an NVIDIA GPU, install the CUDA version of llama-cpp-python for faster inference:
> `CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python`

### 3. Feature Configuration (The Wizard)

Run the granular setup tool. It will generate a secure `.env` file for you.

```bash
python main.py --setup
```

#### Wizard Walkthrough:

1.  **Hardware Profile**: Select `cpu_only` if you lack a GPU, or `medium_vram` if you have 6GB+.
2.  **Solana RPC**:
    *   **Public**: `https://api.mainnet-beta.solana.com` (Rate limited)
    *   **Private**: `https://mainnet.helius-rpc.com/?api-key=...` (Recommended for DeGen Mob)
3.  **Keys**: Enter keys only for the services you plan to use. Leave others blank.

### 4. Local Model Management

cosmos uses Ollama for local intelligence.

```bash
# 1. Install Ollama from ollama.ai

# 2. Pull the Swarm
ollama pull mistral           # General reasoning
ollama pull qwen2.5-coder     # Coding specialist
ollama pull nomic-embed-text  # Embeddings (Crucial for Memory)
```

---

## 🐳 Docker Deployment (Isolated)

For maximum isolation, use Docker. This keeps your environment clean.

```bash
# Basic CPU run
docker-compose -f docker/docker-compose.yml up -d

# GPU accelerated run (requires nvidia-container-toolkit)
docker-compose -f docker/docker-compose.yml --profile gpu up -d
```

Access the dashboard at `http://localhost:8501`.

---

## 🔍 Verification

To verify everything is working locally:

1.  **Check Memory**: `python main.py --cli` -> type `status`. Should show "Memory System: OK".
2.  **Check Solana**: If DeGen mob is enabled, logs will show `Connected to Solana RPC`.
3.  **Check Keys**: Inspect your `.env` file (it is `.gitignore`'d by default) to ensure keys are saved correctly.

---

*"I trust this setup guide is sufficiently foolproof. If not, I can always build a robot to read it for you."*
