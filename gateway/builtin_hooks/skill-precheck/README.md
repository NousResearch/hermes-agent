# Skill Pre-check Hook (SRA) — 技能运行时推荐系统

> **Skill Runtime Advisor (SRA)**: 让 Hermes Agent 知道自己有什么能力，以及什么时候该用什么能力。

---

## 一、这个 Hook 是什么？

当你向 Hermes 发送消息时，这个 Hook 会在消息到达 AI 模型 **之前** 拦截它，用轻量级算法扫描所有已安装的 Skills，自动匹配最相关的 1-2 个技能，并把推荐信息注入到你的消息头部。

**效果示例：**
```
[System Note: Skill Runtime Advisor Recommendations]
Based on your input, the following skills are relevant.
Review them before executing to avoid reinventing the wheel.

⭐ web-access (Score: 44.7, medium confidence)
   -> 所有联网操作必须通过此 skill 处理，包括：搜索、网页抓取...
   -> Match reasons: 同义词'搜索'→'search', name部分'web'

[SRA Processing: 33ms]
---

你的原始消息内容...
```

AI 模型会看到这个系统注记，从而优先加载并使用匹配到的 Skill，避免"重新发明轮子"。

---

## 二、完整架构

```
~/.hermes/hooks/
├── skill-precheck/                  # Hook 本体
│   ├── HOOK.yaml                    # Hook 元数据（事件注册）
│   ├── handler.py                   # 入口：拦截 → 匹配 → 注入
│   ├── README.md                    # 本文件
│   └── __pycache__/                 # Python 缓存
├── core/                            # SRA 核心引擎（共享库）
│   ├── __init__.py
│   ├── advisor.py                   # SkillAdvisor — 推荐引擎主类
│   ├── indexer.py                   # SkillIndexer — 扫描 ~/.hermes/skills/ 建索引
│   ├── matcher.py                   # SkillMatcher — 关键词 + 同义词 + 语义评分
│   ├── memory.py                    # SceneMemory — 使用统计与场景模式记忆
│   └── synonyms.py                  # SYNONYMS — 同义词扩展表
└── data/                            # 运行时数据
    ├── skill_full_index.json        # 技能全文索引（自动构建）
    └── skill_usage_stats.json       # 推荐/使用统计
```

### 数据流

```
用户消息 → Gateway 收到 → emit("agent:pre_process", {message})
                                    ↓
                        handler.py 拦截
                                    ↓
                    SkillAdvisor.recommend(message)
                                    ↓
              indexer → matcher → 评分排序 → top-k 推荐
                                    ↓
              格式化 [System Note: ...] 注记
                                    ↓
              return {"message_override": "注记\n\n原始消息"}
                                    ↓
                    Gateway 替换用户消息
                                    ↓
                    AI 模型接收增强后的消息
```

---

## 三、配置文件

### `HOOK.yaml`
```yaml
name: "skill-precheck"
description: "Intercepts user messages before the Agent processes them to recommend relevant skills via SRA matching."
events:
  - "agent:pre_process"
```

### `~/.hermes/config.yaml` 中的注册
```yaml
hooks:
  skill-precheck:
    path: ~/.hermes/hooks/skill-precheck
```

---

## 四、核心机制详解

### 4.1 索引构建 (indexer.py)
- 扫描 `~/.hermes/skills/` 目录下所有 SKILL.md
- 提取 `triggers`、`description`、`name`、`category` 等元数据
- 构建倒排索引，存入 `data/skill_full_index.json`
- **自动重建**：每 60 秒检查一次，新安装的 skill 会自动被索引

### 4.2 匹配算法 (matcher.py)
评分维度包括：
| 维度 | 说明 |
|------|------|
| 关键词匹配 | 用户输入与 skill triggers/name 的字面重合度 |
| 同义词扩展 | `synonyms.py` 定义的语义等价词（如"搜索"→"search"） |
| 描述相似度 | 与 skill description 的文本相似度 |
| 使用历史 | memory.py 记录的高频场景加权 |

**阈值：**
- `THRESHOLD_STRONG = 80`：强推荐（high confidence）
- `THRESHOLD_WEAK = 50`：弱推荐（medium confidence）
- 低于 50 分不推荐，不会产生任何延迟

### 4.3 快速过滤 (handler.py)
为保证零感知延迟，以下消息直接跳过匹配：
- 空消息
- 以 `/` 开头的 slash 命令
- 长度 < 4 个字符的短消息

### 4.4 容错设计
- 匹配失败或超时 → 静默返回 `None`，**绝不阻塞用户消息**
- 索引损坏 → 自动重建
- Hook 未注册 → Agent 正常运行，只是没有 SRA 推荐

---

## 五、使用方式

### 5.1 日常使用
**无需任何额外操作。** Hook 注册后自动生效，每次发消息时后台自动匹配。

你在消息顶部看到的灰色 `[System Note: Skill Runtime Advisor Recommendations]` 就是它的输出。

### 5.2 为什么有时候看不到推荐？
以下情况不会显示推荐注记：
1. **没有匹配的 skill** — 输入太通用，所有 skill 评分都低于 40 分
2. **短消息 / slash 命令** — 被快速过滤跳过
3. **Feishu 渲染** — 系统注记以灰色小字显示在消息头部，容易被忽略

### 5.3 查看统计数据
```bash
cat ~/.hermes/hooks/data/skill_usage_stats.json
```

字段说明：
- `total_recommendations` — 累计推荐次数
- `skills` — 各 skill 被推荐的次数
- `scene_patterns` — 学到的场景模式

---

## 六、维护与管理

### 6.1 查看 Hook 是否加载
```bash
# 检查 gateway 启动日志
grep "hooks" ~/.hermes/logs/gateway.log | grep -i "load\|discover\|skill-precheck"

# 或直接看日志中的 SRA 标记
grep "Skill Runtime Advisor\|SRA Processing" ~/.hermes/logs/*.log
```

### 6.2 强制刷新索引
索引每 60 秒自动检查更新。如需立即刷新（例如刚安装了新 skill）：
```bash
# 方法 1：重启 gateway（最彻底）
hermes gateway restart

# 方法 2：等待 60 秒，Handler 的 _get_advisor() 会自动重建
```

### 6.3 调试模式
在 handler.py 中临时加入日志输出：
```python
# 在 handle() 函数开头加一行
print(f"[SRA DEBUG] Query: {message!r}, Advisor: {advisor is not None}", flush=True)
```
然后在 `~/.hermes/logs/gateway.log` 中查看。

---

## 七、Hermes 更新后的迁移指南

### 7.1 小版本更新（patch/minor）
**通常无需操作。** Hook 是外部插件，不随 Hermes 核心更新而改变。

更新后验证：
```bash
hermes gateway restart
# 发一条测试消息，检查是否出现 [System Note: Skill Runtime Advisor Recommendations]
```

### 7.2 大版本更新（major）
如果 Hermes 的 Hook 系统 API 发生变化：

1. **检查 `HOOK.yaml` 格式是否仍然兼容**
   ```bash
   cat ~/.hermes/hooks/skill-precheck/HOOK.yaml
   ```
   确认 `events` 字段中的事件名（`agent:pre_process`）是否仍被支持。
   参考 Hermes 官方文档或 `ADDING_A_HOOK.md`（如果有）。

2. **检查 `handle()` 函数签名**
   ```bash
   cat ~/.hermes/hooks/skill-precheck/handler.py
   ```
   确认函数签名是否仍为 `async def handle(event_type, context)`。
   确认返回值格式 `{"message_override": "..."}` 是否仍然有效。

3. **检查 `core/` 模块的导入路径**
   ```bash
   # 确认 Python 路径是否正确
   ls ~/.hermes/hooks/core/
   ```

4. **如果 Hook 系统已废弃**
   Hermes 可能将 SRA 功能内建。检查更新日志中是否提到：
   - `skill_advisor` 或 `SRA` 内置功能
   - `hooks` API 的 breaking changes

### 7.3 迁移 Checklist
- [ ] `hermes gateway restart` 无报错
- [ ] Gateway 日志中无 `[hooks]` 相关 error
- [ ] 发测试消息能看到 `[System Note: Skill Runtime Advisor Recommendations]`
- [ ] `~/.hermes/config.yaml` 中 hooks 配置未被覆盖
- [ ] `~/.hermes/hooks/` 目录完整（skill-precheck/ + core/ + data/）

### 7.4 如果更新后 Hook 不工作了
```bash
# 1. 确认目录还在
ls -la ~/.hermes/hooks/skill-precheck/handler.py

# 2. 确认 config 没被覆盖
grep -A2 "skill-precheck" ~/.hermes/config.yaml

# 3. 手动测试 hook 加载
python3 -c "
import sys; sys.path.insert(0, '/home/jiang/.hermes/hooks')
from skill_precheck.handler import handle
import asyncio
result = asyncio.run(handle('agent:pre_process', {'message': '帮我搜索一个网站'}))
print(result)
"

# 4. 检查 gateway 日志
tail -100 ~/.hermes/logs/gateway.log | grep -i "hook\|sra\|pre_process"
```

---

## 八、常见问题

### Q: SRA 会增加多少延迟？
通常 **30-50ms**，对于 100 个 skill 的索引。快速过滤的消息（短消息、命令）延迟为 0。

### Q: 能自定义推荐阈值吗？
可以。修改 `~/.hermes/hooks/core/advisor.py` 中的：
```python
THRESHOLD_STRONG = 80  # 调高 = 更严格，调低 = 更宽松
THRESHOLD_WEAK = 40
```

### Q: 索引文件可以删除吗？
可以。删除后下次调用时会自动重建：
```bash
rm ~/.hermes/hooks/data/skill_full_index.json
```

### Q: 为什么有些 skill 从未被推荐？
- 该 skill 的 `triggers` 列表为空或太模糊
- 同义词表中缺少对应的映射
- 使用 `advisor.analyze_coverage()` 可以检查覆盖率

### Q: 想让 SRA 推荐更多 skill 怎么办？
修改 `handler.py` 中的 `top_k` 参数：
```python
result = advisor.recommend(message, top_k=2)  # 改成 3 或 5
```

---

## 九、技术细节

- **运行环境**: Python 3.11+, 无外部依赖（纯 stdlib）
- **索引格式**: JSON（`skill_full_index.json`）
- **事件系统**: Hermes Gateway `agent:pre_process` hook
- **线程模型**: 同步调用（handler.py 的 `handle()` 是 async 但实际是 CPU-bound 轻量计算）
- **缓存策略**: Handler 级 60 秒 TTL 缓存，避免每次消息重建索引

---

*最后更新: 2026-05-05 | 版本: 1.0 | 作者: 皓文 & 小智*
