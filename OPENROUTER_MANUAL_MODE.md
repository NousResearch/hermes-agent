# ✅ 已调整：允许手动调用 OpenRouter 模型

**调整时间**：2026-05-04  
**调整内容**：恢复 OPENROUTER_API_KEY，允许手动使用付费模型  
**防护状态**：4 层防护全部保留，防止自动调用

---

## 📊 现在的配置

### ✅ 已恢复
- **OPENROUTER_API_KEY**：已恢复到 ~/.hermes/.env
- **效果**：模型选择菜单中会显示 OpenRouter 目录

### ✅ 仍然保持
- **fallback_providers: []** → 防止自动切换
- **8 个 auxiliary 任务用本地模型** → 防止辅助功能自动调用
- **OpenRouter 每周 $5.00 成本限额** → 硬性成本保护

---

## 🎯 使用方式

### 1️⃣ 启动 Hermes
```bash
hermes chat
```

### 2️⃣ 打开模型选择菜单
在 Hermes 聊天中输入：
```
/model
```

### 3️⃣ 浏览并选择 OpenRouter 模型

**模型列表会显示**：
```
可用模型：
  1. 本地模型
     - qwen3.6:27b (custom)
     - qwen2.5:7b (custom)
     
  2. OpenRouter
     - openrouter/gpt-4-turbo
     - openrouter/claude-opus-4
     - openrouter/gemini-pro
     - ... 其他付费模型
```

### 4️⃣ 选择模型后使用
```
Select model [1-10]: 7  # 选择 OpenRouter 模型
Model switched to: openrouter/gpt-4-turbo

> 现在用高端模型继续对话
```

---

## 🛡️ 防护机制仍然有效

### ✅ 防止自动调用（Layer 2-3）
```yaml
# ~/.hermes/config.yaml
fallback_providers: []  # ← 不自动切换

auxiliary:
  vision:
    provider: custom    # ← 辅助功能用本地
  web_extract:
    provider: custom    # ← 辅助功能用本地
  # ... 其他 auxiliary 都是 custom
```

**结果**：
- ❌ 主模型故障时不自动切换到 OpenRouter
- ❌ 图片分析、网页提取等不调用 OpenRouter
- ✅ 成本：$0（除非手动使用）

### ✅ 成本硬性限制（Layer 4）
- 平台：OpenRouter 账户设置
- 限额：每周 $5.00
- 超出后：自动拒绝请求

---

## 📋 行为对比

| 操作 | 修正前 | 修正后 |
|------|--------|--------|
| **自动调用** | ✓ 可能 | ❌ 不可能 |
| **手动调用** | ✓ 可以 | ✓ 可以 |
| **Fallback** | ✓ OpenRouter | ❌ 禁用 |
| **Auxiliary** | ✓ 自动选择 | ✓ 本地 Ollama |
| **成本限额** | 无 | ✓ $5/周 |
| **预期成本** | $?/周 | $0 + 手动使用 |

---

## 🔍 验证配置

### 确认 OPENROUTER_API_KEY 已恢复
```bash
grep "^OPENROUTER_API_KEY=" ~/.hermes/.env
# 应该显示：OPENROUTER_API_KEY=sk-or-v1-...
```

### 确认 fallback 仍然禁用
```bash
grep "fallback_providers: \[\]" ~/.hermes/config.yaml
# 应该显示：fallback_providers: []
```

### 确认 auxiliary 仍然是本地
```bash
grep -c "provider: custom" ~/.hermes/config.yaml
# 应该显示：9
```

---

## 💰 成本管理

### 预期成本分布

| 场景 | 成本 | 备注 |
|------|------|------|
| 正常使用（默认本地模型） | $0 | 无任何调用 |
| 手动切换到 GPT-4（一次） | $0.01-0.05 | 取决于对话长度 |
| 每周手动使用 OpenRouter | $1-5 | 在 $5/周 限额内 |
| 自动调用（防护启用） | $0 | 不会发生 |

### 监控使用
```bash
# 查看 OpenRouter 使用情况
# 访问：https://openrouter.ai/account/usage

# 查看实时日志
tail -f ~/.hermes/logs/agent.log | grep -i "openrouter"
```

---

## ⚠️ 关键提醒

### 以下情况 still protected
- ✅ 主模型故障 → 不自动切换
- ✅ 网页提取 → 用本地模型
- ✅ 图片分析 → 用本地模型
- ✅ 其他辅助功能 → 用本地模型

### 以下情况需要手动操作
- 如需使用 GPT-4：`/model` 选择 openrouter/gpt-4-turbo
- 如需使用 Claude：`/model` 选择 openrouter/claude-opus
- 使用后可随时切换回本地模型

### 成本限制
- 每周最多 $5.00（硬性上限）
- 超出后 OpenRouter 自动拒绝请求
- 不会产生隐藏或意外费用

---

## 🔄 随时可调整

### 如需再次禁用 OpenRouter
```bash
sed -i 's/^OPENROUTER_API_KEY=/#OPENROUTER_API_KEY=/' ~/.hermes/.env
```

### 如需恢复完整 fallback
```yaml
# 添加到 ~/.hermes/config.yaml
fallback_providers:
  - provider: openrouter
    model: openrouter/gpt-4-turbo
```

### 如需临时禁用某个 auxiliary
```yaml
# 在 ~/.hermes/config.yaml 中改为 disabled
web_extract:
  enabled: false  # ← 禁用这个功能
```

---

## ✅ 总结

**现在的配置**：
- ✅ 自动调用：完全禁止（fallback + auxiliary 本地）
- ✅ 手动调用：完全可用（API key 已恢复）
- ✅ 成本保护：始终有效（$5/周 限额）

**日常使用**：
1. 默认使用本地 Qwen3.6 模型（$0 成本）
2. 需要高端模型时，`/model` 选择 OpenRouter
3. 成本永远不会超过 $5/周

**风险等级**：LOW ✓  
**功能完整性**：100% ✓  
**预期成本**：$0（+ 手动使用的 OpenRouter 费用）

---

*配置调整完成于 2026-05-04*  
*状态：准备就绪，可正常使用*
