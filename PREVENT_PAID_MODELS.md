# Hermes 防止自动调用 OpenRouter 付费模型配置指南

## 问题

Hermes 有时会在主模型失败、超时或达到速率限制时自动切换到其他模型，包括 OpenRouter 的付费模型。这会导致意外的成本。

## 解决方案

有 3 种策略可防止自动调用付费模型，从最简单到最灵活：

---

## 策略 1：完全禁用自动切换（推荐用于本地模型）

**问题特征**：您只想用本地 Ollama 模型，不希望任何自动切换。

**配置**：
```yaml
# ~/.hermes/config.yaml

model:
  default: qwen3.6:27b
  provider: custom
  base_url: http://172.22.144.1:11434/v1
  context_length: 262144
  ollama_num_ctx: 262144

# ← 删除或注释掉 fallback_model 和 fallback_providers
# 这样当主模型失败时，Hermes 会错误退出而不是切换模型
```

**优点**：
- 最简单
- 完全控制，零意外成本
- 当模型失败时可以立即看到错误

**缺点**：
- 模型故障时无自动恢复

---

## 策略 2：配置免费的本地 fallback（推荐用于需要容错）

**问题特征**：您想要容错，但只使用本地或完全免费的模型。

**配置**：
```yaml
# ~/.hermes/config.yaml

model:
  default: qwen3.6:27b
  provider: custom
  base_url: http://172.22.144.1:11434/v1
  context_length: 262144

# 只在主模型失败时切换到另一个本地模型
# （Ollama 中必须已加载）
fallback_providers:
  - provider: custom
    model: qwen2.5:7b
    base_url: http://172.22.144.1:11434/v1
  - provider: custom
    model: gemma4:26b
    base_url: http://172.22.144.1:11434/v1
```

**优点**：
- 有自动容错
- 零成本（全部本地）
- 完全可控

**缺点**：
- 需要多个模型已加载

---

## 策略 3：配置只使用免费的 OpenRouter 模型

**问题特征**：您想使用 OpenRouter，但只想用免费或按量付费模型，避免高端付费模型。

**配置**：
```yaml
# ~/.hermes/config.yaml

model:
  default: qwen3.6:27b
  provider: custom
  base_url: http://172.22.144.1:11434/v1
  context_length: 262144

# 配置本地 fallback + OpenRouter 免费模型
fallback_providers:
  # 首先尝试本地模型
  - provider: custom
    model: qwen2.5:7b
    base_url: http://172.22.144.1:11434/v1
  
  # 然后尝试 OpenRouter 的免费模型（无需额外 API key，按消费计费）
  # ⚠️ 注意：这些模型在 OpenRouter 可能有使用配额或定价，请确认费率
  - provider: openrouter
    model: openrouter/meta-llama/llama-2-70b-chat  # 免费限额模型示例

# 配置 OpenRouter 提供商限制：忽略所有高端付费模型
# 只允许特定的免费/便宜模型
providers:
  openrouter:
    # 只允许这些 providers 处理您的请求（OpenRouter 的后端）
    # 这些是成本较低或提供免费层级的 provider
    # 参考：https://openrouter.ai/docs#performance-and-routing
    allowed_providers:
      - "Fireworks"
      - "Together"
      - "Lepton"
      - "Azure"
    
    # 忽略这些高成本 providers
    ignore_providers:
      - "Anthropic"        # ← 高成本
      - "OpenAI"           # ← 高成本
      - "Google"           # ← 中等成本
      - "Perplexity"       # ← 中等成本
      - "xAI"              # ← 中等成本
```

**优点**：
- 使用 OpenRouter 的灵活性
- 可以手动指定费率限制
- 仍然有自动容错

**缺点**：
- 需要管理允许列表/忽略列表
- OpenRouter 定价会随时间变化

---

## 策略 4：手动切换模型（最安全）

如果您想在每次使用前手动选择模型（零自动切换风险）：

```bash
# 启动 Hermes 并立即切换模型
hermes chat

# 然后在 Hermes 中使用
/model                    # 打开交互式模型选择器
# 或
/model --set custom:qwen3.6:27b  # 显式设置为本地模型
```

这样任何模型切换都需要您的明确同意。

---

## 推荐配置（为您而言）

基于您的环境（本地 Ollama + RTX 3090），我推荐**策略 1**：

```yaml
# ~/.hermes/config.yaml

model:
  default: qwen3.6:27b
  provider: custom
  api_key: '5114624'
  base_url: http://172.22.144.1:11434/v1
  context_length: 262144
  ollama_num_ctx: 262144

providers: {}

# ← 完全删除 fallback_model 和 fallback_providers
fallback_providers: []

credential_pool_strategies: {}
toolsets:
  - hermes-cli
```

**好处**：
- ✅ 绝不会意外调用付费模型
- ✅ 完全透明（故障时看到清晰错误）
- ✅ 零成本
- ✅ 最简单的配置

---

## 验证当前配置

```bash
# 查看当前 fallback 配置
cat ~/.hermes/config.yaml | grep -A 5 "fallback"

# 查看允许/忽略的 provider
cat ~/.hermes/config.yaml | grep -A 10 "providers:"

# 测试模型（会显示是否有 fallback）
hermes -v 2>&1 | grep -i "fallback\|provider"
```

---

## 如果 Hermes 已在使用 OpenRouter 怎么办？

如果您发现 Hermes 正在调用 OpenRouter（检查 `~/.hermes/logs/agent.log`），立即：

```bash
# 1. 停止 gateway
hermes --stop-gateway

# 2. 清除 fallback 配置
cat ~/.hermes/config.yaml | grep -v "fallback" > /tmp/config.yaml
cp /tmp/config.yaml ~/.hermes/config.yaml

# 3. 重启
hermes --restart-gateway
```

然后检查日志以确认不再使用 OpenRouter：

```bash
tail -f ~/.hermes/logs/agent.log | grep -i "openrouter\|provider\|billing"
```

---

## 相关配置选项详解

### fallback_model / fallback_providers

```yaml
# 旧格式（单个 fallback）
fallback_model:
  provider: openrouter
  model: openrouter/meta-llama/llama-2-70b

# 新格式（链式 fallback，按顺序尝试）
fallback_providers:
  - provider: custom
    model: local-model-1
  - provider: custom
    model: local-model-2
  - provider: openrouter
    model: cheap-model  # 最后的选择
```

### providers_ignored 和 providers_allowed

```yaml
# 在 AIAgent 初始化时由 CLI 读取
# CLI 选项：
#   hermes --provider-ignore openai,anthropic
#   hermes --provider-only fireworks,together

# 配置文件暂无对应选项（需通过 CLI flag 或代码传递）
```

---

## 成本控制清单

- [ ] 确认 `fallback_providers` 为空或只包含本地模型
- [ ] 确认主模型不是 OpenRouter（或者如果是，已配置 `providers_ignored`）
- [ ] 查看 OpenRouter 账户，确保没有启用代理模型
- [ ] 设置 OpenRouter 支出上限：https://openrouter.ai/account/billing/limits
- [ ] 定期检查 `~/.hermes/logs/agent.log` 中的模型使用情况

---

## 命令快速参考

```bash
# 显示当前配置
hermes config show | grep -A 5 "model:\|fallback\|provider"

# 设置配置值
hermes config set model.default qwen3.6:27b
hermes config set fallback_providers []

# 查看模型使用日志
tail -100 ~/.hermes/logs/agent.log | grep -E "model=|provider=|routing"

# 测试模型连接（不实际调用，只检查可达性）
hermes --model qwen3.6:27b --test

# 禁用特定 provider（一次性）
hermes --provider-ignore openai,anthropic,google chat
```

---

## FAQ

**Q: 为什么 Hermes 会自动切换模型？**  
A: 当主模型：
- 返回 503/429/429 等速率限制/过载错误
- 返回 402/billing 等账单错误
- 超时或连接断开
- 返回 context_length_exceeded 错误

此时 Hermes 会尝试 fallback 模型。

**Q: 如何知道 Hermes 是否在使用付费模型？**  
A: 检查日志：
```bash
grep "provider=" ~/.hermes/logs/agent.log | tail -20
```

查找非 "custom" 的 provider（如 "openrouter", "openai", "anthropic"）。

**Q: 我不小心启用了 OpenRouter fallback，已经产生了成本怎么办？**  
A: 
1. 立即禁用 OpenRouter：删除或空置 `fallback_providers`
2. 检查 OpenRouter 账户和使用情况
3. 如果是额外成本，联系 OpenRouter 支持
4. 在 https://openrouter.ai/account/billing/limits 设置支出上限

**Q: 如何只在紧急情况下使用 OpenRouter fallback？**  
A: 
```bash
# 默认只用本地模型
# 需要付费 fallback 时，使用：
hermes --fallback-provider openrouter:gpt-4-turbo chat
```

---

更新日期：2026-05-04  
这份指南适用于 Hermes v0.12.0+
