---
name: mmx-cli
description: Use mmx (MiniMax CLI) to generate text, images, video, speech, and music via MiniMax AI platform. Use when the user wants to create media content, chat with MiniMax models, perform web search, or generate audio. Install via `npm install -g mmx-cli`. Already configured on this system with MiniMax CN API key.
---

# MiniMax CLI — mmx

MiniMax 多模态生成 CLI 工具。认证已配置（`~/.mmx/config.json`），本机已安装。

## Agent 使用标志（必须）

| 标志 | 用途 |
|------|------|
| `--non-interactive` | 缺少参数时立即失败，不提示 |
| `--quiet` | 抑制进度动画，stdout 纯数据 |
| `--output json` | 机器可读 JSON 输出 |
| `--yes` | 跳过确认提示 |

## 已验证可用的命令

### text chat — 文字对话
```bash
mmx text chat --message "<text>" --output json --quiet --non-interactive
# 示例
mmx text chat --message "用一句话介绍自己" --output json --quiet --non-interactive
```

### image generate — 图片生成
```bash
mmx image generate --prompt "<描述>" --output json --quiet --non-interactive [--aspect-ratio 16:9] [--n 2]
# 示例
mmx image generate --prompt "赛博朋克城市夜景" --output json --quiet --non-interactive
# 输出：图片文件名（如 image_001.jpg），保存在当前目录
```

### video generate — 视频生成（异步）
```bash
mmx video generate --prompt "<描述>" --output json --quiet --non-interactive --async
# 示例（--async 返回任务 ID，稍后查询结果）
mmx video generate --prompt "机器人跳舞" --output json --quiet --non-interactive --async
```

### speech synthesize — 语音合成
```bash
mmx speech synthesize --text "<文本>" --output json --quiet --non-interactive [--language en|zh] [--speed 1.0]
# 示例
mmx speech synthesize --text "你好，我是 MiniMax AI" --language zh --output json --quiet --non-interactive
```

### music generate — 音乐生成
```bash
mmx music generate --prompt "<描述>" [--lyrics "<歌词>"] --output json --quiet --non-interactive
# 示例
mmx music generate --prompt "阳光明媚的日子" --lyrics "我爱阳光" --output json --quiet --non-interactive
```

### search — 网页搜索
```bash
mmx search query --text "<查询内容>" --output json --quiet --non-interactive
# 示例
mmx search query --text "MiniMax AI 最新消息" --output json --quiet --non-interactive
```

## 认证状态
```bash
mmx auth status
```
认证文件：`~/.mmx/config.json`（region=cn），`~/.mmx/credentials.json`（api_key）

## 重要说明

- **图片保存位置**：默认保存在执行命令时的当前工作目录
- **异步任务**：视频/音乐生成使用 `--async` 返回任务 ID
- **区域**：`mmx` 自动检测，也可 `--region global` 或 `--region cn` 覆盖
- **pip 安装版本 vs npm 安装版本**：npm 版本（`mmx-cli`）是最新版，功能更全
