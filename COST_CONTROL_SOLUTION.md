# 🎯 Hermes 成本控制完整方案

> **创建时间**：2026-05-04  
> **问题**：Hermes 自动调用 OpenRouter 付费模型导致成本超支  
> **解决方案**：禁用自动 fallback，完全控制模型选择

---

## 📋 您的情况

- **主模型**：本地 Qwen3.6 27B（Ollama）
- **环境**：WSL2 RTX 3090 24GB
- **问题**：Hermes 在主模型失败时自动切换到 OpenRouter 付费模型
- **目标**：零成本，完全本地模型

---

## ⚡ 立即修复（3 分钟）

### 方式 1：自动脚本（最简单）

```bash
bash scripts/disable-paid-models.sh
# 选择选项 1：禁用所有 fallback
```

### 方式 2：一行命令

```bash
cp config-no-fallback.yaml ~/.hermes/config.yaml
```

### 方式 3：手动编辑

```bash
nano ~/.hermes/config.yaml

# 找到这些行，删除或注释：
#   fallback_model: ...
#   fallback_providers: ...

# 或替换为：
#   fallback_providers: []
```

### 验证

```bash
# 应该显示：fallback_providers: []
grep fallback_providers ~/.hermes/config.yaml

# 重启 Hermes
hermes chat
```

---

## 📚 详细文档

| 文件 | 用途 |
|------|------|
| [QUICK_FIX_PAID_MODELS.md](./QUICK_FIX_PAID_MODELS.md) | **首先阅读** - 快速修复指南 |
| [PREVENT_PAID_MODELS.md](./PREVENT_PAID_MODELS.md) | 4 种完整策略对比 |
| [config-no-fallback.yaml](./config-no-fallback.yaml) | 推荐配置模板 |
| [scripts/disable-paid-models.sh](./scripts/disable-paid-models.sh) | 交互式配置脚本 |
| [scripts/audit-paid-models.sh](./scripts/audit-paid-models.sh) | 审计工具 - 检查当前风险 |

---

## 🚀 推荐流程

### 第 1 步：了解风险

```bash
bash scripts/audit-paid-models.sh
```

输出示例：
```
✅ Fallback 状态：已禁用
✅ OpenRouter 配置：无
✅ 最近使用：本地模型
✅ API Keys：未配置

风险等级：LOW
```

### 第 2 步：阅读推荐

```bash
cat QUICK_FIX_PAID_MODELS.md
```

5 分钟快速理解为什么会自动调用付费模型

### 第 3 步：应用配置

```bash
# 备份原配置
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.backup.$(date +%s)

# 应用推荐配置
cp config-no-fallback.yaml ~/.hermes/config.yaml
```

### 第 4 步：验证

```bash
# 检查配置
hermes config show | grep -A 3 "fallback"

# 测试
hermes chat

# 查看日志确认无 OpenRouter 调用
tail -20 ~/.hermes/logs/agent.log | grep -i provider
```

---

## 🎯 4 种解决方案对比

### 1️⃣ 完全禁用 fallback（⭐ 推荐）

**配置**：
```yaml
fallback_providers: []
```

**优点**：
- ✅ 最简单
- ✅ 零成本
- ✅ 完全安全
- ✅ 适合纯本地模型

**缺点**：
- ❌ 主模型故障时无自动恢复

**使用场景**：
- 本地开发
- 成本敏感
- 绝不想有意外费用

---

### 2️⃣ 本地 fallback

**配置**：
```yaml
fallback_providers:
  - provider: custom
    model: qwen2.5:7b
    base_url: http://172.22.144.1:11434/v1
  - provider: custom
    model: gemma4:26b
    base_url: http://172.22.144.1:11434/v1
```

**优点**：
- ✅ 有自动容错
- ✅ 零成本（全本地）
- ✅ 完全可控

**缺点**：
- ❌ 需要多个模型已加载

**使用场景**：
- 需要高可用
- 多个本地模型可用

---

### 3️⃣ 免费 OpenRouter fallback

**配置**：
```yaml
fallback_providers:
  - provider: custom
    model: qwen2.5:7b
  - provider: openrouter
    model: openrouter/meta-llama/llama-2-70b
```

**优点**：
- ✅ 灵活的备选方案
- ✅ 可以使用新模型

**缺点**：
- ❌ ⚠️ 可能产生成本
- ❌ 需要管理 API key
- ❌ OpenRouter 定价变化

**使用场景**：
- 需要高级功能的 fallback
- 能够监控和控制成本

**安全措施**：
- 设置 OpenRouter 支出限额：https://openrouter.ai/account/billing/limits
- 定期检查使用情况

---

### 4️⃣ 完全手动（最安全）

**每次使用前手动选择**：
```bash
hermes chat
/model --set custom:qwen3.6:27b
```

**优点**：
- ✅ 绝对安全
- ✅ 完全可控
- ✅ 零自动化风险

**缺点**：
- ❌ 需要每次手动指定

**使用场景**：
- 高风险环境
- 需要完全透明

---

## ❓ FAQ

### Q：为什么 Hermes 会自动调用 OpenRouter？

A：因为配置了 `fallback_providers` 或 `fallback_model`。当主模型：
- 速率限制（429）
- 过载（503）
- 账单问题（402）
- 超时
- 上下文长度超限

Hermes 会自动切换到 fallback 模型。

### Q：如何知道 Hermes 是否在使用 OpenRouter？

A：检查日志：
```bash
tail -50 ~/.hermes/logs/agent.log | grep -i "openrouter\|provider="
```

寻找非 "custom" 的 provider。

### Q：已经产生了成本怎么办？

A：
1. 立即停用 OpenRouter：
   ```bash
   sed -i 's/fallback_providers:.*/fallback_providers: []/g' ~/.hermes/config.yaml
   ```

2. 检查账单：
   - OpenRouter：https://openrouter.ai/account/usage
   - OpenAI：https://platform.openai.com/account/billing/usage
   - Anthropic：https://console.anthropic.com/account/usage

3. 设置限额防止未来成本：
   - OpenRouter：https://openrouter.ai/account/billing/limits

4. 联系支持寻求退款（如额度过高）

### Q：如何设置支出限额？

A：对于 OpenRouter：
1. 登录 https://openrouter.ai/account/billing/limits
2. 设置 "Hard Limit" - 超过该额度会自动拒绝请求
3. 或设置 "Monthly Limit" - 月度预算

### Q：可以同时使用多个本地模型吗？

A：可以，配置 fallback_providers：
```yaml
fallback_providers:
  - provider: custom
    model: qwen2.5:7b
    base_url: http://172.22.144.1:11434/v1
  - provider: custom
    model: gemma4:26b
    base_url: http://172.22.144.1:11434/v1
```

### Q：如何恢复原配置？

A：
```bash
cp ~/.hermes/config.yaml.backup ~/.hermes/config.yaml
# 或
git checkout ~/.hermes/config.yaml  # 如果配置在 git 中
```

---

## 🔧 命令速查表

```bash
# 查看当前 fallback 配置
grep -A 5 "fallback" ~/.hermes/config.yaml

# 禁用 fallback
sed -i 's/fallback_providers:.*/fallback_providers: []/g' ~/.hermes/config.yaml

# 查看所有 provider 配置
hermes config show | grep -i provider

# 实时查看模型使用
tail -f ~/.hermes/logs/agent.log | grep -i "provider="

# 审计当前风险
bash scripts/audit-paid-models.sh

# 一键应用推荐配置
cp config-no-fallback.yaml ~/.hermes/config.yaml

# 启动配置向导
bash scripts/disable-paid-models.sh
```

---

## 📊 成本预防矩阵

| 措施 | 成本影响 | 操作难度 | 推荐度 |
|------|---------|---------|--------|
| 禁用 fallback | $0 永久 | ⭐ 非常简单 | ⭐⭐⭐⭐⭐ |
| 本地 fallback | $0 永久 | ⭐⭐ 简单 | ⭐⭐⭐⭐ |
| 设置 API 限额 | $X 有上限 | ⭐ 非常简单 | ⭐⭐⭐ |
| 监控日志 | $0 | ⭐⭐ 简单 | ⭐⭐⭐ |
| 手动切换模型 | $0 永久 | ⭐⭐⭐ 中等 | ⭐⭐ |

---

## 🎓 下一步

1. **现在**：选择方案 1（禁用 fallback）应用配置
2. **5 分钟**：阅读 QUICK_FIX_PAID_MODELS.md 理解为什么
3. **10 分钟**：运行审计脚本验证配置
4. **可选**：查看 PREVENT_PAID_MODELS.md 了解其他方案

---

## 📞 需要帮助？

- **快速修复**：查看 QUICK_FIX_PAID_MODELS.md
- **完整指南**：查看 PREVENT_PAID_MODELS.md
- **故障排查**：运行 `bash scripts/audit-paid-models.sh`
- **应急处理**：查看 FAQ "已产生成本怎么办"

---

**推荐操作顺序**：
```
1. bash scripts/audit-paid-models.sh              # 了解当前状态
2. cat QUICK_FIX_PAID_MODELS.md                    # 快速理解
3. cp config-no-fallback.yaml ~/.hermes/config.yaml # 应用配置
4. grep fallback_providers ~/.hermes/config.yaml    # 验证
5. hermes chat                                      # 测试
```

**预计时间**：15 分钟内完全解决

---

*创建于 2026-05-04 - Hermes v0.12.0+*
