---
name: memory-refine
description: Use when 自动提炼 Hermes 会话到 fact_store(增量,LLM)。触发:记忆自动提炼/memory-refine。
version: 1.0.0
author: BruceAi66
license: MIT
dependencies: [python3]
platforms: [linux, macos]
metadata:
  hermes:
    tags: [memory, fact_store, holographic, LLM, cron, automation, CJK]

---

# Hermes Memory Refine

从 Hermes 会话记录(`state.db`)增量提炼「关于用户的稳定观察」,写入 holographic
`fact_store`(L2 记忆)。增量锚点保证**只提炼新消息**,历史内容不会反复重提,避免
LLM token 浪费。这是对 Hermes 预置记忆系统的**补充**(主动提炼),不是替代。

## When to Use

- 想要 Hermes「自动」从日常对话中提炼用户偏好/行为模式/项目状态,写入 fact_store
- 希望每天/定时运行,而不是靠人工逐条写记忆
- 有 Hermes + holographic memory(默认) + 一个 OpenAI 兼容 LLM 端点

## 原理

```
state.db (messages表, role=user)
   │ 增量锚点 last_msg_id (只取新消息)
   ▼
LLM (OpenAI 兼容 /chat/completions) 提炼稳定观察
   │
   ▼
2-gram 去重预检 (中文友好, 与已有 facts 比对)
   │
   ▼
MemoryStore.add_fact (官方入口: 实体/HRR向量/FTS 全自动)
```

去重双层保险:
1. **2-gram 语义预检**:观察内容拆成 2 字片段,与已有 facts 比对,命中率高即跳过
   (为什么不用 FTS5 预检?FTS5 按空格分词,中文整段算一个 token,`search_facts`
   对中文自然语言查询命中率为 0 —— 这是中文召回的老坑,见 PR #97050 的 trigram 修复)
2. **content UNIQUE 约束**:`add_fact` 自带,精确重复自动返回旧 id

## 安装

```bash
# 复制脚本到 Hermes scripts 目录
mkdir -p ~/.hermes/scripts
cp scripts/memory_refine.py ~/.hermes/scripts/

# 配置环境变量(加到 shell profile 或 cron 环境)
export HERMES_HOME="$HOME/.hermes"                      # Hermes 数据目录(默认 ~/.hermes)
export HERMES_LLM_API_KEY="sk-..."                      # OpenAI 兼容端点 key(必填)
export HERMES_LLM_BASE_URL="https://api.openai.com/v1"  # 端点(默认 OpenAI)
export HERMES_LLM_MODEL="gpt-4o-mini"                   # 模型(默认 gpt-4o-mini)
# 若 Hermes 装在非标准路径,再加:
# export HERMES_INSTALL_DIR="/path/to/hermes-agent"
```

## 用法

```bash
# 冒烟:只打印提炼结果(不写库,不推进锚点)
python3 ~/.hermes/scripts/memory_refine.py --hours 24

# 正式:增量提炼 + 写库 + 推进锚点
python3 ~/.hermes/scripts/memory_refine.py --hours 24 --write

# cron:全被去重挡掉时静默退出(no_agent watchdog 模式)
python3 ~/.hermes/scripts/memory_refine.py --write --quiet

# 调试:忽略锚点全量回看
python3 ~/.hermes/scripts/memory_refine.py --reset-anchor --write
```

参数:
- `--hours N`:回看上限(小时)。增量锚点存在时只取新消息;锚点无(首次)时回看 N 小时
- `--write`:写库(否则 dry-run 只打印)
- `--quiet`:全被去重挡掉时零输出(配 cron no_agent,空输出=无变化)
- `--reset-anchor`:忽略已有锚点全量回看(调试用)

## Cron 定时

每天 06:00 自动跑(no_agent 纯脚本,不耗 agent token):

```bash
# 建议加到 Hermes cron(在 Hermes 里):script = "memory_refine.py --write --quiet", no_agent = true
# 或系统 crontab:
0 6 * * * cd ~ && HERMES_HOME=... HERMES_LLM_API_KEY=... python3 ~/.hermes/scripts/memory_refine.py --write --quiet
```

增量语义:
- 提炼成功(无论新增或全去重)→ 推进锚点 `last_msg_id`
- 提炼失败/解析空 → **不推进**,下次重试同一批(不丢消息)
- 锚点存在时只处理新消息 → 历史内容永不重提,零 token 浪费

## 输出格式

脚本调用 LLM 提炼,要求输出每行一条:
```
类型|确定性|内容
```
类型 ∈ {偏好, 行为模式, 关注主题, 项目状态, 环境事实, 个人背景}
确定性 ∈ {确定, 推断}

写入 fact_store 时类型映射:偏好/行为模式/个人背景→user_pref,项目状态→project,其余→general。tag 统一 `memory_refine`(可追溯来源)。

## Pitfalls

- **中文去重**:FTS5 按空格分词对中文无效,2-gram 预检是必须的(踩过坑)
- **reasoning 模型**:deepseek 等推理模型 content 偶发为空(先烧 reasoning),
  脚本有重试兜底;`max_tokens` 默认 4000 别调太小
- **写库必须走 MemoryStore 官方入口**:直接 SQL INSERT 会绕过 HRR 向量/FTS/实体解析
- **CJK 查询召回**:本脚本的去重依赖读取已有 facts,若中文检索本身有问题,
  配合 PR #97050 的 trigram 修复一起用效果最好
- **认证**:OpenAI 兼容端点一律 `Authorization: Bearer <key>`;某些端点(如 Portal)
  模型名要带 `~` 前缀,用 HERMES_LLM_MODEL 配置即可
