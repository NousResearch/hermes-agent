# /offer：秋招进度管理

把零散的招聘邮件、截图和聊天记录整理成可复盘的求职漏斗，记录公司、岗位、日期、状态、下一步和备注。不得把申请编号、密码、验证码或无关个人信息写入公开资源。

## 资源定位

ASu 资源支持两种布局，按以下顺序定位：Claude Code 安装布局使用 `../../assets/asu/` 与 `../../references/asu/`；仓库插件布局使用 `../../assets/` 与 `../../references/`。每个候选均为一对 assets/references 目录：仅当 assets 目录同时包含 `application-tracker.html` 和 `application-tracker-overview.svg`，且对应 references 目录包含 `email-monitoring.md` 时才使用该目录对，避免误用其他 skill 的资源。`application-tracker.html` 是可编辑的求职进度表，`application-tracker-overview.svg` 是预览图。

如果 skill 被单独复制到其他目录，先从当前 skill 目录向上依次定位 `assets/asu/`、`references/asu/` 与仓库插件的 `assets/`、`references/`，不要重新制作已有资源。

## 工作流程

1. 从用户提供的邮件、招聘网站或截图中提取日期、公司、岗位、当前状态、下一步和必要备注。
2. 合并同一公司同一岗位的重复记录；后来的测评、面试、拒信或 Offer 状态覆盖早期的“已投递”。
3. 只在有证据时更新状态；普通自动回执不能推断为面试或 Offer，证据不足标记为“待确认”。
4. 用户未指定文件位置时，将 `application-tracker.html` 复制到桌面，并返回生成文件的绝对路径。
5. 用户要求查看邮箱或每日检查时，使用浏览器处理；登录、密码、MFA、验证码和 CAPTCHA 由用户完成，不代为保存。
6. 用户需要同时制作简历文件时，转入 `/resume`；需要改写经历时，转入 `/asu`。

## 状态规范

默认状态包括：已投递、筛选中、测评中、面试、Offer、拒绝/已结束、待确认。每次更新尽量写明证据来源和下一步动作，便于后续复盘。

## 默认交付

默认提供进度表文件、已识别记录摘要、状态变化、下一步清单和缺失信息。不要把用户的真实求职记录写进 skill 的模板或 README。
