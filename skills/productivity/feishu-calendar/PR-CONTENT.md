# 飞书日历技能 - GitHub PR 提交内容

## 📋 PR 标题

```
Add Feishu Calendar Skill - 飞书日历管理技能
```

## 📝 PR 描述

```markdown
## 📋 概述

添加飞书（Feishu/Lark）日历管理技能，支持创建、查询、删除日程，使用 OAuth 2.0 进行身份验证，Token 自动刷新。

## ✨ 功能特性

- **创建日程**: 支持标题、时间、描述
- **查询日程**: 按日期查询，返回 JSON 格式
- **删除日程**: 通过 event_id 删除
- **OAuth 2.0**: 使用 user_access_token，事件同步到用户个人日历
- **自动刷新**: access_token 2 小时自动刷新，refresh_token 30 天有效期
- **配置检查**: setup.py 提供配置状态检查和引导

## 📁 文件结构

```
skills/productivity/feishu-calendar/
├── SKILL.md                      # 技能主文档（配置说明 + 使用指南）
├── PUBLISHING.md                 # 发布指南（内部使用）
├── scripts/
│   ├── setup.py                  # 配置检查脚本
│   ├── feishu_oauth.py           # OAuth 2.0 管理（生成链接、刷新 token）
│   └── feishu_calendar.py        # 日历 API 操作（创建/查询/删除）
└── references/
    └── oauth-setup.md            # OAuth 配置详细指南
```

## 🔧 配置要求

用户需要：
1. 飞书开放平台应用（App ID + App Secret）
2. 配置环境变量：`FEISHU_APP_ID`, `FEISHU_APP_SECRET`, `FEISHU_REDIRECT_URI`
3. 完成 OAuth 2.0 授权（首次使用，约 2 分钟）

## 🚀 使用方法

```bash
# 配置检查
python3 scripts/setup.py --check

# OAuth 授权（首次使用）
python3 scripts/feishu_oauth.py generate_link
python3 scripts/feishu_oauth.py exchange <code>

# 创建日程
python3 scripts/feishu_calendar.py create --title "会议" --start "2026-04-16 09:00" --end "2026-04-16 10:00"

# 查询日程
python3 scripts/feishu_calendar.py list "2026-04-16"

# 删除日程
python3 scripts/feishu_calendar.py delete "event_id_xxx"
```

## ✅ 测试验证

- [x] 创建日程成功
- [x] 查询日程成功
- [x] 删除日程成功
- [x] Token 自动刷新机制
- [x] 飞书 App 同步验证（事件可见）
- [x] 配置检查脚本
- [x] 错误处理完善
- [x] 文档完整性

## 🔒 安全说明

- Token 本地存储（`~/.hermes/.feishu_user_token.json`），权限 600
- 仅申请必要的日历权限（`calendar:calendar`）
- 无硬编码凭证（App ID/Secret 通过环境变量配置）
- 已清理所有敏感信息（无真实 App ID、Token、Code）
- OAuth 流程符合飞书开放平台规范

## 📚 相关文档

- [飞书开放平台 - 日历 API](https://open.feishu.cn/document/server-docs/calendar-service)
- [飞书开放平台 - OAuth 2.0](https://open.feishu.cn/document/ukTMukTMukTM/uEjNwUjL2YDM14SM2ATN)

## 🏷️ 技能标签

`Feishu`, `Lark`, `Calendar`, `日历`, `OAuth`, `日程管理`, `productivity`

## 📝 许可证

MIT License
```

## 💾 Commit Message

```
feat: add Feishu Calendar skill for personal calendar management

- Add feishu-calendar skill to productivity category
- Implement OAuth 2.0 authentication with auto-refresh
- Support create/list/delete calendar events
- Use user_access_token for personal calendar integration
- Add setup.py for configuration checking
- Include comprehensive documentation (SKILL.md, oauth-setup.md)
- Token auto-refresh: access_token (2h), refresh_token (30d)
- Security: no hardcoded credentials, local token storage (chmod 600)

Resolves: Personal calendar integration for Hermes Agent
```

## 🌿 Git 分支名

```
feat/feishu-calendar-skill
```

## 📦 发布步骤

### 1. Fork 仓库
访问 https://github.com/NousResearch/hermes-agent 并点击 Fork

### 2. 克隆仓库
```bash
git clone https://github.com/<your-username>/hermes-agent.git
cd hermes-agent
```

### 3. 创建分支
```bash
git checkout -b feat/feishu-calendar-skill
```

### 4. 复制技能文件
```bash
cp -r ~/.hermes/skills/productivity/feishu-calendar \
      skills/productivity/
```

### 5. 提交更改
```bash
git add skills/productivity/feishu-calendar
git commit -m "feat: add Feishu Calendar skill for personal calendar management"
```

### 6. 推送到远程
```bash
git push origin feat/feishu-calendar-skill
```

### 7. 创建 Pull Request
1. 访问 https://github.com/NousResearch/hermes-agent/compare
2. 选择你的分支 `feat/feishu-calendar-skill`
3. 使用上面的 **PR 标题** 和 **PR 描述**
4. 点击 "Create Pull Request"
5. 等待 Hermes 团队审核合并

## 📬 备用方案：直接分享压缩包

如果不想通过 PR 发布，可以分享压缩包到社区：

```bash
# 压缩包已生成
ls -lh ~/.hermes/skills/productivity/feishu-calendar-skill.tar.gz
```

分享到：
- Hermes Agent Discord 服务器
- Hermes Agent Telegram 群
- GitHub Discussions
- 相关社区论坛

## ✅ 发布前检查清单

- [x] 所有敏感信息已清理（App ID、App Secret、Token、Code）
- [x] SKILL.md 文档完整
- [x] 配置指南清晰
- [x] 脚本无硬编码凭证
- [x] 使用环境变量或配置文件
- [x] 错误处理完善
- [x] 有故障排查指南
- [x] 通过新用户安装测试

---

**生成时间**: 2026-04-15  
**技能版本**: 1.0.0  
**许可证**: MIT
