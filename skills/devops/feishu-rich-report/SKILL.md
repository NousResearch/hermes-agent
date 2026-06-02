---
name: feishu-rich-report
version: 1.0.0
description: "重要汇报用 lark-cli 发飞书富文本（post），支持 Markdown 表格。Hermes 内置回复会把表格降级成纯 text，本 skill 不改 Hermes 源码。当用户要求向飞书群汇报结果、且内容含表格/多级标题/结构化总结时使用。"
metadata:
  requires:
    bins: ["lark-cli"]
  tags: [feishu, lark-cli, report, markdown, table]
---

# Feishu 富文本汇报（lark-cli）

> **前置：** 阅读 `~/.claude/skills/lark-shared/SKILL.md`（认证与 `--as`）。

## 何时用本 skill（必守）

| 场景 | 用什么 |
|------|--------|
| 进度条、单行状态、`⚙️ mcp_...` | Hermes **正常回复**即可 |
| **重要汇报**：测试结论、验收、含 **Markdown 表格** | **本 skill** → `feishu_rich_send.py` |
| 向指定群汇报（用户给了 `oc_xxx`） | `--chat-id oc_xxx` |
| 在当前飞书会话里跟帖 | 不设 chat-id，用 `HERMES_SESSION_CHAT_ID` + 可选 `--thread-id` |

**禁止**对含表格的终稿使用 Hermes `send_message` / 普通聊天回复（会变成 `msg_type=text`，表格不渲染）。

## 一键发送

```bash
# 当前 Hermes 飞书会话（环境变量由 gateway 注入）
python3 skills/devops/feishu-rich-report/scripts/feishu_rich_send.py \
  --markdown $'## 标题\n\n| 项 | 结果 |\n| --- | --- |\n| A | 通过 |'

# 指定群 + 从文件读（长报告）
python3 skills/devops/feishu-rich-report/scripts/feishu_rich_send.py \
  --chat-id oc_bfa70445b7411381db594886a2201495 \
  --markdown-file /path/to/REPORT.md

# 回复到用户消息所在话题
python3 skills/devops/feishu-rich-report/scripts/feishu_rich_send.py \
  --thread-id om_xxx \
  --markdown-file ./summary.md

# 预览不发送
python3 skills/devops/feishu-rich-report/scripts/feishu_rich_send.py \
  --chat-id oc_xxx --markdown $'## test' --dry-run
```

工作目录：在 `hermes-agent` 仓库根执行，或写脚本的绝对路径。

## 与 lark-cli 直接调用等价

```bash
lark-cli im +messages-send --as bot \
  --chat-id oc_xxx \
  --markdown $'## ...\n\n| a | b |\n|---|---|\n| 1 | 2 |'
```

`feishu_rich_send.py` 封装了：环境变量 chat/thread、长度校验、错误 JSON 解析。

## 撰写建议

- 用 `##` / `###` 分段，用 `|` 表格汇总结果
- 超长正文（>24k 字符）：飞书文档 + 消息里只发摘要表 + 文档链接
- 需要卡片/按钮：改用 `lark-cli` `interactive` 或 `lark-im` skill，不在此脚本范围

## 参考

- Hermes 为何无表格：`gateway/platforms/feishu.py` 中 `_build_outbound_payload` 对表格强制 `text`
- 看板短消息仍用：`kanban-feishu-live/scripts/kanban_feishu_stage_notify.py`（`--text`）
