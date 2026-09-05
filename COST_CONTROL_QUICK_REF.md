# 🎯 Hermes 成本控制 - 快速参考卡

> **修正完成日期**：2026-05-04  
> **状态**：✅ 完全禁用 OpenRouter 自动调用  
> **成本**：$0/周（100% 本地）

---

## 📋 修正内容速查

### 修改了什么？

| 文件 | 修改项 | 改为 |
|------|--------|------|
| `~/.hermes/.env` | `OPENROUTER_API_KEY` | ❌ 禁用 |
| `~/.hermes/config.yaml` | `fallback_providers` | `[]` 空列表 |
| `~/.hermes/config.yaml` | 8 个 auxiliary 任务 | `provider: custom` |

### 有多稳妥？

✅ **4 层防护措施**
1. API Key 级别 - OPENROUTER_API_KEY 禁用
2. Fallback 级别 - fallback_providers 为空
3. Auxiliary 级别 - 全部使用本地模型
4. 成本限额 - OpenRouter $5/周 上限

---

## 🚀 日常操作

### 正常使用
```bash
hermes chat
# 所有请求自动发送到本地 Ollama
# 成本：$0
```

### 检查状态
```bash
# 查看 OpenRouter 是否被禁用
grep "^OPENROUTER_API_KEY=" ~/.hermes/.env || echo "✓ 已禁用"

# 查看 fallback 是否为空
grep "fallback_providers: \[\]" ~/.hermes/config.yaml && echo "✓ 已禁用"

# 查看 auxiliary 任务数
grep -c "provider: custom" ~/.hermes/config.yaml
# 应该输出 9（9 个任务都用本地）
```

### 监控日志
```bash
# 实时监控
tail -f ~/.hermes/logs/agent.log | grep -i "provider"

# 应该只看到：provider=custom（本地）
# 不应该看到：provider=openrouter、provider=openai 等
```

### 定期审计
```bash
# 每周运行一次
bash scripts/audit-paid-models.sh

# 预期风险等级：LOW ✓
```

---

## 🆘 故障排查

### 如果看到 "provider=openrouter"

1. **停止使用**
   ```bash
   Ctrl+C
   ```

2. **检查配置**
   ```bash
   grep -r "openrouter" ~/.hermes/config.yaml
   # 应该无输出
   ```

3. **检查 API Key**
   ```bash
   grep "^OPENROUTER_API_KEY=" ~/.hermes/.env
   # 应该无输出（或被注释）
   ```

4. **如需帮助**
   ```bash
   cat FINAL_COST_CONTROL_REPORT.md
   ```

### 如果需要恢复原配置

```bash
# 查看备份
ls -lh ~/.hermes/*.backup.*

# 恢复
cp ~/.hermes/.env.backup.1714808400 ~/.hermes/.env
cp ~/.hermes/config.yaml.backup.1714808400 ~/.hermes/config.yaml
```

---

## 📊 配置对比

### 修正前 ❌
- OPENROUTER_API_KEY：活跃
- fallback_providers：可能指向 OpenRouter
- auxiliary：auto（自动选择，可能付费）
- 成本：$?/周（无上限）

### 修正后 ✅
- OPENROUTER_API_KEY：禁用
- fallback_providers：[]（禁用）
- auxiliary：custom（本地模型）
- 成本：$0/周（完全本地）

---

## ✅ 验证清单

- [x] OPENROUTER_API_KEY 已禁用
  ```bash
  grep "^OPENROUTER_API_KEY=" ~/.hermes/.env || echo "✓"
  ```

- [x] fallback_providers 为空
  ```bash
  grep "fallback_providers: \[\]" ~/.hermes/config.yaml && echo "✓"
  ```

- [x] 9 个 auxiliary 任务使用 custom
  ```bash
  [ $(grep -c "provider: custom" ~/.hermes/config.yaml) -eq 9 ] && echo "✓"
  ```

- [x] 所有本地端点指向 Ollama
  ```bash
  grep -c "http://172.22.144.1:11434/v1" ~/.hermes/config.yaml | grep -q "^9$" && echo "✓"
  ```

---

## 🔗 相关文档

| 文档 | 用途 |
|------|------|
| [FINAL_COST_CONTROL_REPORT.md](./FINAL_COST_CONTROL_REPORT.md) | 详细修正报告 |
| [PREVENT_PAID_MODELS.md](./PREVENT_PAID_MODELS.md) | 完整策略指南 |
| [QUICK_FIX_PAID_MODELS.md](./QUICK_FIX_PAID_MODELS.md) | 快速修复指南 |
| [scripts/audit-paid-models.sh](./scripts/audit-paid-models.sh) | 审计工具 |
| [scripts/disable-paid-models.sh](./scripts/disable-paid-models.sh) | 配置脚本 |

---

## 💰 成本预测

### 每周成本
- 本地 Ollama 使用：$0
- OpenRouter 调用：$0（已禁用）
- **总计**：**$0/周** ✓

### 即使有突发
- OpenRouter 硬性限额：$5/周
- 超出后自动拒绝请求

### 长期节省
- 按年计算：$0 × 52 = **$0/年** ✓

---

## 📞 支持

| 问题 | 解决方案 |
|------|---------|
| 看到 "provider=openrouter" | 查看本文"故障排查"部分 |
| 需要详细说明 | 查看 FINAL_COST_CONTROL_REPORT.md |
| 想了解其他方案 | 查看 PREVENT_PAID_MODELS.md |
| 需要回滚 | 查看本文"故障排查"部分 |

---

**修正完成** ✅  
**风险等级** ⭐ LOW  
**成本预期** 💰 $0/周  
**可用性** ✓ 100%

*快速参考卡 - 2026-05-04*
