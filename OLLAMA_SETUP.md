# Ollama 配置指南 - Context Length 256K

## 问题
Ollama 默认 context window 为 2048 tokens，导致 Hermes 使用时 context 严重不足。

## 解决方案

### Windows 侧 (E:\Ollama)

在 Windows 的 `run_ollama.bat` 或启动 Ollama 时，设置环境变量：

```batch
@echo off
set OLLAMA_CONTEXT_LENGTH=262144
start ollama serve
```

或者在 PowerShell 中：

```powershell
$env:OLLAMA_CONTEXT_LENGTH = 262144
ollama serve
```

### 验证配置

启动后，运行以下命令检查 qwen3.6:27b 模型的 context 大小：

```bash
curl http://172.22.144.1:11434/api/show -X POST -d '{"name":"qwen3.6:27b"}' | jq '.parameters'
```

预期输出应该包含：
```
num_ctx     262144
```

### Hermes 侧配置

已在 `~/.hermes/config.yaml` 中设置：

```yaml
model:
  default: qwen3.6:27b
  provider: custom
  base_url: http://172.22.144.1:11434/v1
  context_length: 262144
  ollama_num_ctx: 262144  # 显式告诉 Hermes 使用 262144 context
```

## 验证 Hermes 识别

启动 Hermes 时应看到日志：

```
✓ Config context_length loaded: 262144 tokens
📊 Context limit: 262,144 tokens (compress at 50% = 131,072)
```

## 常见问题

1. **仍然显示 131,072 tokens?**
   - 检查 Ollama 是否以 `OLLAMA_CONTEXT_LENGTH=262144` 启动
   - 检查 `~/.hermes/config.yaml` 中 `context_length: 262144` 是否存在

2. **Ollama 无法分配这么多内存?**
   - 降低 context length，例如 128K：`OLLAMA_CONTEXT_LENGTH=131072`
   - 或在 config.yaml 中相应调整

3. **模型加载失败?**
   - 检查 RTX 3090 VRAM 是否足够（262144 context ≈ 12-15GB）
   - 使用量化版本（Q4_K_M）而不是更高精度
