# Hermes Agent on Raspberry Pi 2 (ARMv7 32-bit)
## Installation and Operation Manual

---

## Table of Contents
1. [Hardware and System Requirements](#hardware-and-system-requirements)
2. [Installation Steps](#installation-steps)
3. [Using lm-studio (Recommended)](#using-lm-studio-recommended)
4. [Using llama.cpp (Advanced)](#using-llamacpp-advanced)
5. [FAQ](#faq)

---

## Hardware and System Requirements

| Item | Specification |
|------|---------------|
| Device | Raspberry Pi 2 Model B |
| CPU | ARMv7 900MHz (4 cores) |
| RAM | 1GB LPDDR2 |
| Storage | microSD card (at least 8GB) |
| OS | Raspberry Pi OS Lite (32-bit) |

**Note:** Raspberry Pi 2 uses a 32-bit ARM architecture and does not support 64-bit software.

---

## Installation Steps

### 1. Update the system

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git wget
```

### 2. Create a virtual environment

```bash
python3 -m venv ~/.hermes-venv
source ~/.hermes-venv/bin/activate
pip install --upgrade pip
```

### 3. Install Hermes Agent dependencies

```bash
pip install honcho-ai sentence-transformers pypdf beautifulsoup4
```

---

## Using lm-studio (Recommended)

### Advantages
- ✅ Graphical interface, easy to operate
- ✅ No need to compile llama.cpp
- ✅ Supports hot-swapping models
- ✅ Built-in model download support

### Disadvantages
- ⚠️ Requires a graphical interface (desktop environment)

### Install lm-studio

```bash
# Download lm-studio for Linux
wget https://github.com/lmstudio-ai/lmstudio/releases/download/v0.4.0/lmstudio_0.4.0_amd64.deb
sudo dpkg -i lmstudio_0.4.0_amd64.deb
```

**Note:** Raspberry Pi 2 is ARMv7, so lm-studio must be compiled for ARM or used through a web version.

### Alternative: Use a Web API (Recommended)

#### 1. Install the llama.cpp Web Server (ARMv7)

```bash
# Clone llama.cpp
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Compile (disable AVX/AVX2, enable NEON)
make clean
make -j$(nproc) LLAMA_AVX2=OFF LLAMA_AVX=OFF LLAMA_F16C=OFF LLAMA_FMA=OFF server

cd ..
```

#### 2. Download the Qwen2.5-7B GGUF model

```bash
# Install huggingface-cli
pip install huggingface_hub

# Download the model (q4_k_m quantized version, about 4.6GB)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir ~
```

#### 3. Start the llama.cpp service

```bash
cd ~/llama.cpp
./server -m ~/qwen2.5-7b-instruct-q4_k_m.gguf \
    -c 2048 \
    --port 8080 \
    -np 1 \
    --n_gpu_layers 0
```

**Parameter notes:**
- `-c 2048`: context size (limited by Raspberry Pi 2 RAM)
- `--n_gpu_layers 0`: do not use GPU (Raspberry Pi 2 has no CUDA)

#### 4. Test the API

```bash
curl http://localhost:8080/v1/models
```

Expected response:
```json
{
  "data": [
    {
      "id": "qwen2.5-7b-instruct-q4_k_m",
      "object": "model"
    }
  ]
}
```

#### 5. Configure Hermes Agent

```bash
# Create the config file
cat > ~/.hermes/config.yaml << 'EOF'
models:
  - name: "custom:qwen2.5-7b"
    base_url: "http://localhost:8080/v1"
    api_key: "not-used"

providers:
  - name: "custom"
    provider: "openai_compatible"
    base_url: "http://localhost:8080/v1"
    api_key: "not-used"

memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: "honcho"
  nudge_interval: 10
  flush_min_turns: 6

tools:
  - memory
  - skills
  - terminal
  - session_search
EOF
```

#### 6. Install Hermes Agent

```bash
git clone https://github.com/matttest0080-prog/hermes-agent-iot.git
cd hermes-agent-iot
pip install -e .
```

#### 7. Start Hermes Agent

```bash
source ~/.hermes-venv/bin/activate
hermes
```

---

## Using llama.cpp (Advanced)

### Advantages
- ✅ Pure command line, no graphical interface required
- ✅ Better resource control
- ✅ Supports GPU acceleration (not useful on Raspberry Pi 2)

### Disadvantages
- ⚠️ Requires compilation (about 15-30 minutes)
- ⚠️ Requires manual model management

### Installation Steps

#### 1. Compile llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp

# Compile (ARMv7 32-bit)
make clean
make -j$(nproc) LLAMA_AVX2=OFF LLAMA_AVX=OFF LLAMA_F16C=OFF LLAMA_FMA=OFF

# Test the build
./llama-cli --version
```

#### 2. Download the model

```bash
# Use huggingface-cli
pip install huggingface_hub

# Download Qwen2.5-7B (q4_k_m quantized)
huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
    qwen2.5-7b-instruct-q4_k_m.gguf \
    --local-dir ~
```

#### 3. Start the server

```bash
cd ~/llama.cpp
./server -m ~/qwen2.5-7b-instruct-q4_k_m.gguf \
    -c 2048 \
    --port 8080 \
    -np 1
```

#### 4. Test

```bash
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "qwen2.5-7b-instruct-q4_k_m",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7
    }'
```

---

## FAQ

### Q1: Not enough memory when compiling llama.cpp?

**Solution:** Use fewer cores.
```bash
make -j2 LLAMA_AVX2=OFF LLAMA_AVX=OFF  # Use only 2 cores
```

### Q2: Model loading failed?

**Reason:** The GGUF model is too large (4.6GB) and exceeds available RAM.
**Solution:**
- Use a smaller model, such as `qwen2.5-1.5B`
- Use the `--n_ctx` parameter to reduce context size

```bash
./server -m ~/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 1024 --port 8080
```

### Q3: Hermes shows `no module named honcho-ai` on startup?

**Solution:**
```bash
source ~/.hermes-venv/bin/activate
pip install honcho-ai sentence-transformers pypdf beautifulsoup4
```

### Q4: How do I configure memory limits?

Adjust `~/.hermes/config.yaml`:
```yaml
memory:
  memory_char_limit: 1500   # Reduce memory usage
  user_char_limit: 800
```

---

## Quick Start Scripts

### start_lm_server.sh
```bash
#!/bin/bash
# Start the llama.cpp service

cd ~/llama.cpp
./server -m ~/qwen2.5-7b-instruct-q4_k_m.gguf \
    -c 2048 \
    --port 8080 \
    -np 1 \
    --n_gpu_layers 0
```

### start_hermes.sh
```bash
#!/bin/bash
# Start Hermes Agent

cd ~/hermes-agent-iot
source ~/.hermes-venv/bin/activate
hermes
```

### Usage
```bash
# Terminal 1 - start the model service
chmod +x start_lm_server.sh
./start_lm_server.sh

# Terminal 2 - start Hermes
chmod +x start_hermes.sh
./start_hermes.sh
```

---

## Verification Test

In the Hermes CLI, enter:
```
/hermes tools list
```

Expected output:
```
✓ enabled  memory          💾 Memory
✓ enabled  skills          🛠 Skills
✓ enabled  terminal        🖥 Terminal
✓ enabled  session_search  🔍 Session Search
```

---

## Notes

1. **Raspberry Pi 2 is 32-bit ARM**, so make sure all software supports ARMv7.
2. **Memory is limited** (1GB), so avoid running multiple models at the same time.
3. **Use quantized models** (q4_k_m, q5_k_m) to reduce memory usage.
4. **Use a virtual environment** to avoid conflicts with the system Python.

---

**Version**: v0.2-pi2  \
**Author**: Matt0080828  \
**Repository**: https://github.com/matttest0080-prog/hermes-agent-iot

---

## Quick Migration Script (New Machine)

### One-click migration to a new Raspberry Pi 2

```bash
#!/bin/bash
# setup_pi2_hermes.sh - Quick setup for Raspberry Pi 2

set -e

echo "=== Hermes Agent Quick Setup for Pi2 ==="

# Step 1: Update system
echo "Step 1: Updating system..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential git wget

# Step 2: Clone hermes-agent (shallow)
echo "Step 2: Cloning hermes-agent (shallow)..."
git clone --depth=1 https://github.com/matttest0080-prog/hermes-agent-iot.git ~/hermes-agent-iot

# Step 3: Create virtual environment
echo "Step 3: Creating virtual environment..."
python3 -m venv ~/.hermes-venv
source ~/.hermes-venv/bin/activate

# Step 4: Install hermes-agent
echo "Step 4: Installing hermes-agent..."
cd ~/hermes-agent-iot
pip install -e .

# Step 5: Install RAG dependencies
echo "Step 5: Installing RAG dependencies..."
pip install honcho-ai sentence-transformers pypdf beautifulsoup4

# Step 6: Setup config
echo "Step 6: Creating config..."
mkdir -p ~/.hermes
cat > ~/.hermes/config.yaml << 'EOF'
models:
  - name: "custom:qwen2.5-7b"
    base_url: "http://localhost:8080/v1"
    api_key: "not-used"

providers:
  - name: "custom"
    provider: "openai_compatible"
    base_url: "http://localhost:8080/v1"
    api_key: "not-used"

memory:
  memory_enabled: true
  user_profile_enabled: true
  memory_char_limit: 2200
  user_char_limit: 1375
  provider: "honcho"
  nudge_interval: 10
  flush_min_turns: 6

tools:
  - memory
  - skills
  - terminal
  - session_search
EOF

echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Build llama.cpp: cd ~/ && git clone https://github.com/ggerganov/llama.cpp.git && cd llama.cpp && make -j\$(nproc) LLAMA_AVX2=OFF LLAMA_AVX=OFF"
echo "2. Download model: huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF qwen2.5-7b-instruct-q4_k_m.gguf --local-dir ~"
echo "3. Start server: cd ~/llama.cpp && ./server -m ~/qwen2.5-7b-instruct-q4_k_m.gguf -c 2048 --port 8080"
echo "4. Run hermes: cd ~/hermes-agent-iot && source ~/.hermes-venv/bin/activate && hermes"
```

### Memory optimization settings (1GB RAM)

If system memory is insufficient, adjust `~/.hermes/config.yaml`:

```yaml
memory:
  memory_char_limit: 1000   # Reduce memory usage
  user_char_limit: 500
```

### Use a smaller model (512MB RAM)

```bash
# Download a smaller model
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \
    qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --local-dir ~

# Start the service
./server -m ~/qwen2.5-1.5b-instruct-q4_k_m.gguf -c 1024 --port 8080
```
