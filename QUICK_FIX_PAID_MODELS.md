# 立即应用：禁用 OpenRouter 自动付费模型

## ⚡ 快速解决方案（3 步）

您遇到的问题："hermes agent 有时候会自作主张调用 OpenRouter 的付费模型"

**根本原因**：Hermes 配置了 `fallback_providers` 或 `fallback_model`，当主模型失败时会自动切换。

### 步骤 1：备份当前配置

```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.backup.$(date +%s)
```

### 步骤 2：禁用 fallback（选择一种方案）

#### 方案 A：完全禁用（推荐 ⭐）
最简单、最安全。适合您因为您有本地 Ollama。

```bash
# 选项 1：使用自动脚本
bash scripts/disable-paid-models.sh

# 选项 2：手动配置
cat config-no-fallback.yaml > ~/.hermes/config.yaml

# 选项 3：手动编辑
nano ~/.hermes/config.yaml
# 找到这两行，删除或注释掉：
#   fallback_model: ...
#   fallback_providers: ...
# 或替换为：
#   fallback_providers: []
```

#### 方案 B：只用本地 fallback
如果您想要容错但不使用 OpenRouter。

在 `~/.hermes/config.yaml` 中添加：

```yaml
fallback_providers:
  - provider: custom
    model: qwen2.5:7b
    base_url: http://172.22.144.1:11434/v1
  - provider: custom
    model: gemma4:26b
    base_url: http://172.22.144.1:11434/v1
```

### 步骤 3：验证并测试

```bash
# 查看配置是否已更改
grep -A 2 "fallback" ~/.hermes/config.yaml
# 输出应该是：fallback_providers: [] 或被注释掉

# 测试 Hermes
hermes chat

# 检查日志确认没有 OpenRouter 调用
tail -50 ~/.hermes/logs/agent.log | grep -i "provider\|openrouter\|routing"
```

---

## ✅ 验证清单

- [ ] 已备份原 config.yaml
- [ ] `fallback_providers` 为空或删除
- [ ] `fallback_model` 不存在或被注释
- [ ] 重启后 `hermes chat` 正常工作
- [ ] 日志中没有 "openrouter" 或 "api_key" 字样（只有 "custom" provider）

---

## 🔍 如何确认 Hermes 不会调用 OpenRouter

在启动 Hermes 后，立即检查：

```bash
# 方式 1：查看实时日志
tail -f ~/.hermes/logs/agent.log &
hermes chat

# 寻找这些关键字：
#   ✓ provider=custom         → 使用本地模型（安全）
#   ✗ provider=openrouter    → 使用 OpenRouter（可能产生成本）
#   ✗ api_key=                → 调用远程 API（可能产生成本）

# 方式 2：查看配置
hermes config show | grep -i "provider\|fallback"

# 方式 3：测试模型切换
/model
# 输出应该只显示本地/custom 模型，没有 OpenRouter
```

---

## 🚨 如果已产生成本怎么办

1. **立即停用 fallback**：
   ```bash
   sed -i 's/fallback_providers:.*/fallback_providers: []/g' ~/.hermes/config.yaml
   ```

2. **检查 OpenRouter 账户**：
   - 登录 https://openrouter.ai/account/usage
   - 查看最近的 API 调用和费用

3. **设置支出上限**（防止未来成本）：
   - 访问 https://openrouter.ai/account/billing/limits
   - 设置月度支出上限

4. **联系 OpenRouter 支持**：
   - 如果意外费用过高，解释情况可能会退款

---

## 📚 完整文档

详细配置说明见：[PREVENT_PAID_MODELS.md](./PREVENT_PAID_MODELS.md)

其中包括：
- 4 种策略对比
- 每种策略的优缺点
- FAQ 和故障排查
- 本地模型与付费模型的成本对比

---

## 🎯 推荐行动（针对您的环境）

基于您的配置（本地 RTX 3090 + Ollama + 本地 Qwen3.6 27B）：

✅ **做法**：
- 使用本地 Ollama 模型作为主模型
- 禁用所有 fallback（防止自动切换）
- 零成本，完全可控

❌ **不做**：
- 不配置 OpenRouter fallback
- 不设置自动 provider 切换
- 不连接付费 API

**立即执行**：
```bash
# 应用推荐配置
cp config-no-fallback.yaml ~/.hermes/config.yaml

# 验证
grep fallback_providers ~/.hermes/config.yaml
# 输出应该是：fallback_providers: []

# 启动
hermes chat
```

---

## 🔗 相关文件

- [PREVENT_PAID_MODELS.md](./PREVENT_PAID_MODELS.md) - 完整配置指南
- [config-no-fallback.yaml](./config-no-fallback.yaml) - 推荐配置模板
- [scripts/disable-paid-models.sh](./scripts/disable-paid-models.sh) - 自动配置脚本

---

**最后更新**：2026-05-04  
**版本**：Hermes v0.12.0+

有任何问题，查看 `PREVENT_PAID_MODELS.md` FAQ 部分或运行：
```bash
grep -r "fallback" ~/.hermes/logs/ | tail -20
```
