# 🚀 创建 Pull Request 指南

## 📋 快速步骤

### 1️⃣ 访问你的 Fork

打开你的 GitHub 仓库：
```
https://github.com/RichardQidian/hermes-agent
```

### 2️⃣ 切换到分支

确保你在正确的分支：
```
Branch: feature/json-config-system
```

### 3️⃣ 创建 Pull Request

**方法 A: 从你的分支页面**

1. 访问：https://github.com/RichardQidian/hermes-agent/tree/feature/json-config-system
2. 点击 **"Contribute"** 按钮
3. 选择 **"Open pull request"**

**方法 B: 直接访问 PR 创建页面**

访问：
```
https://github.com/RichardQidian/hermes-agent/pull/new/feature/json-config-system
```

### 4️⃣ 配置 PR

**Base repository:** 
- 点击 "base repository" 下拉框
- 选择：`hermes-agent/hermes-agent`

**Base branch:**
- 选择：`main` (或 `develop`，根据项目惯例)

**Title:**
```
feat: Add JSON Configuration System with Centralized Provider Management
```

**Description:**
复制 `docs/config-system/PR_DESCRIPTION_EN.md` 的全部内容

### 5️⃣ 提交 PR

1. 点击 **"Create pull request"** 按钮
2. 等待 CI 检查运行
3. 回应维护者的反馈

---

## 📝 PR 描述模板

以下是完整的英文 PR 描述，直接复制使用：

```markdown
# feat: Add JSON Configuration System with Centralized Provider Management

## 📋 Overview

This PR introduces a modern JSON-based configuration system for Hermes Agent, inspired by the clean design of openclaw.json. The new format addresses critical pain points in the current YAML configuration.

### Problems Solved

1. **API Key Duplication**: No more repeating API keys across multiple configuration sections
2. **Scattered Provider Config**: All provider settings centralized in one location
3. **Complex Structure**: Flatter, more readable structure (2-3 levels vs 4-5)
4. **Hard to Extend**: Adding new models is now as simple as pushing to an array

## 🎯 Key Features

### 1. Centralized Provider Management

```json
{
  "providers": {
    "bailian": {
      "base_url": "https://coding.dashscope.aliyuncs.com/v1",
      "api_key": "${BAILIAN_API_KEY}",
      "models": [
        { "id": "qwen3.5-plus", "name": "Qwen 3.5 Plus" },
        { "id": "qwen3.6-plus", "name": "Qwen 3.6 Plus" },
        { "id": "glm-5", "name": "GLM-5" }
      ]
    }
  }
}
```

### 2. Environment Variable Substitution

```json
{
  "providers": {
    "bailian": { "api_key": "${BAILIAN_API_KEY}" },
    "openrouter": { "api_key": "${OPENROUTER_API_KEY}" }
  },
  "platforms": {
    "feishu": {
      "app_id": "${FEISHU_APP_ID}",
      "app_secret": "${FEISHU_APP_SECRET}"
    }
  }
}
```

### 3. Unified Feature Configuration

```json
{
  "features": {
    "vision": { "provider": "bailian", "model": "qwen3.5-plus" },
    "compression": { "provider": "bailian", "model": "qwen3.5-plus" },
    "session_search": { "provider": "bailian", "model": "text-embedding-v4" }
  }
}
```

## 📁 Files Added

```
hermes-agent/
├── hermes_cli/
│   └── config_json.py              # JSON config loader with env expansion
├── scripts/
│   └── migrate_config.py           # YAML → JSON migration tool
└── docs/
    └── config-system/
        ├── JSON_CONFIG_GUIDE.md    # Comprehensive documentation
        └── PR_DESCRIPTION.md       # This PR description
```

## 🔄 Migration Path

### Automatic Migration

```bash
# Preview migration
hermes config json migrate --dry-run

# Apply migration
hermes config json migrate --apply

# Compare formats
hermes config json migrate --compare
```

### CLI Commands

```bash
# Show current JSON config
hermes config json show

# Migrate with options
hermes config json migrate --apply
hermes config json migrate --compare
hermes config json migrate --dry-run
```

### Backward Compatibility

- ✅ `config.json` takes priority if it exists
- ✅ Falls back to `config.yaml` if JSON not found
- ✅ No breaking changes to existing functionality
- ✅ Users can revert by removing `config.json`

## 📊 Format Comparison

| Aspect | YAML (Legacy) | JSON (New) | Improvement |
|--------|---------------|------------|-------------|
| **Config Lines** | ~327 | ~109 | **-66%** |
| **Provider Config** | Scattered | Centralized | **Maintainable** |
| **API Key Refs** | Multiple | Single (${VAR}) | **More Secure** |
| **Structure Depth** | 4-5 levels | 2-3 levels | **Clearer** |
| **Add New Model** | Edit multiple sections | Push to array | **Simpler** |

## 🧪 Testing

### Unit Tests

```bash
# Test config loading
hermes config json show

# Test migration
hermes config json migrate --dry-run

# Test environment variable expansion
python -c "from hermes_cli.config_json import load_config_json; print(load_config_json())"
```

### Integration Tests

- ✅ Loads existing `config.yaml` user configurations
- ✅ Expands environment variables from `.env` file
- ✅ Migrates all major configuration sections
- ✅ Backward compatible with YAML fallback

## 🔐 Security Considerations

- ✅ API keys stored in `.env` (not in config file)
- ✅ `${VAR}` syntax prevents accidental commits
- ✅ `redact_secrets` continues to apply to JSON config output
- ✅ File permissions remain 0600
- ✅ No hardcoded credentials in configuration

## 📝 Checklist

- [x] Code implementation complete
- [x] Migration tool tested
- [x] Documentation written
- [x] Backward compatibility verified
- [x] Security review (API key handling)
- [x] All tests passing
- [x] CLI integration complete

## 🎉 Impact

This PR makes Hermes Agent configuration:
- **66% smaller** (327 → 109 lines)
- **Easier to understand** (centralized providers)
- **Easier to maintain** (no duplicate API keys)
- **More secure** (environment variable references)
- **Simpler to extend** (add models via array)

---

**Related Issue:** N/A (New feature)  
**Breaking Changes:** None (fully backward compatible)  
**Migration Required:** Optional (automatic migration tool provided)
```

---

## ✅ 提交前检查清单

### 代码质量
- [x] 所有测试通过
- [x] 代码格式化（lint 通过）
- [x] 无语法错误
- [x] 向后兼容

### 文档
- [x] PR 描述完整
- [x] 使用指南编写完成
- [x] 测试报告准备就绪
- [x] 示例代码正确

### Git
- [x] 提交信息清晰
- [x] 分支命名规范
- [x] 已推送到 GitHub
- [x] 无冲突

---

## 🔗 重要链接

### 你的仓库
- **Fork:** https://github.com/RichardQidian/hermes-agent
- **分支:** https://github.com/RichardQidian/hermes-agent/tree/feature/json-config-system
- **Commits:** https://github.com/RichardQidian/hermes-agent/commits/feature/json-config-system

### 官方仓库
- **Hermes Agent:** https://github.com/hermes-agent/hermes-agent
- **官方 PR 列表:** https://github.com/hermes-agent/hermes-agent/pulls

### 文档位置
- **PR 描述（英文）:** `~/hermes-agent/docs/config-system/PR_DESCRIPTION_EN.md`
- **使用指南:** `~/hermes-agent/docs/config-system/JSON_CONFIG_GUIDE.md`
- **测试报告:** `~/hermes-agent/TEST_REPORT.md`
- **实现总结:** `~/hermes-agent/COMPLETION_SUMMARY.md`

---

## 💡 提示

### 1. 等待 CI 检查

提交 PR 后，GitHub Actions 会自动运行测试。确保：
- ✅ 所有检查通过（绿色勾）
- ⚠️ 如果有失败，查看日志并修复

### 2. 回应反馈

维护者可能会提出修改建议：
- 保持礼貌和专业
- 及时回应问题
- 根据需要修改代码

### 3. 更新 PR

如果需要修改：
```bash
# 在本地修改代码
git add .
git commit -m "fix: Address review comments"
git push origin feature/json-config-system
```

PR 会自动更新！

---

## 🎊 准备好了吗？

### 立即创建 PR

点击下面的链接直接创建：

👉 **[Create Pull Request Now](https://github.com/RichardQidian/hermes-agent/pull/new/feature/json-config-system)**

### 或者先预览

1. 访问你的分支：https://github.com/RichardQidian/hermes-agent/tree/feature/json-config-system
2. 查看文件和提交历史
3. 确认无误后再创建 PR

---

## 📞 需要帮助？

如果在创建 PR 过程中遇到问题：

1. **查看 GitHub 文档:** https://docs.github.com/en/pull-requests
2. **检查 PR 指南:** https://docs.github.com/en/pull-requests/collaborating-with-pull-requests
3. **联系维护者:** 在官方仓库中提问

---

**祝你顺利！** 🎉

你的贡献将让 Hermes Agent 变得更好！
