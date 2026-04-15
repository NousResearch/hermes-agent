# 上下文压缩方案二：Cursor 式重构设计笔记

## 问题背景

当前 Hermes 的上下文压缩方案（**方案一已经把 protect_first_n 改成 0**）：
- 仍然是「头部保护 + 中间摘要 + 尾部保护」的策略
- 对于编程场景：对话轮次少，但单轮内容很长
- 问题：当主题漂移/需求演进后，过时的上下文依然可能留在头部占用空间

## 目标

参考 Cursor 的设计思路：
- **完全抛弃「固定头尾」策略**
- **只保留最近 N 轮完整对话 + 前面所有内容的结构化摘要**
- **完整原始历史存档到文件**，需要时可以检索找回

## 设计方案

### 1. 核心压缩逻辑

```
原始消息列表 [M1, M2, M3, ..., Mn]
  ↓
计算每个消息的 token 长度
  ↓
从尾部往前累加 token，直到达到 token 预算（比如 tail_budget = threshold * 20%）
  ↓
超过预算的所有消息 → 统一做结构化摘要
  ↓
结果：[摘要] + [最近 N 轮完整消息]
```

**关键点：**
- **不区分「头部/中间/尾部」**，只区分「历史」vs「最近」
- 初始的 system prompt 怎么办？→ system prompt 本来就不在 conversation messages 里，它一直在最前面，不压缩
- 如果用户初始需求在第一条 user message，超过预算会被摘要进去，但摘要会保留目标信息

### 2. 完整历史存档

每次压缩触发时：
- 在压缩前，把当前完整的原始对话保存到：
  ```
  ~/.hermes/sessions/{session_id}/history/full_{compression_count}.md
  ```
- 文件名：`full_1.md`, `full_2.md`, ...
- 格式：每条消息带上 role 和序号，便于检索

### 3. （可选）增加历史检索工具

给 Agent 添加一个工具：`search_history(query: str)`
- 作用：在存档的完整历史中搜索相关内容
- 实现：可以用 APINexus 的重排序模型来做检索排序
- 触发：当 Agent 发现摘要里信息不够，可以主动调用找回细节

### 4. 与当前方案的对比

| 维度 | 当前方案（改完 protect_first_n=0） | Cursor 方案 |
|-----|-------------------------------|------------|
| 压缩策略 | 滑动窗口（头+中+尾） | 最近 N 轮 + 全历史摘要 |
| 主题漂移问题 | 大幅缓解 | 彻底解决 |
| 压缩效果 | 中等 | 更好（更干净） |
| 实现改动 | 小（已完成） | 较大（需要重构compress方法） |
| 完整历史找回 | ❌ 没有 | ✅ 有 |

### 5. 改造步骤

1. **第一步**：修改 `_find_tail_cut_by_tokens` 方法，去掉 `compress_start = self.protect_first_n` 的硬编码，从头部开始一直到 token 预算截断
2. **第二步**：添加完整历史存档功能，在 `compress()` 方法开头保存原始对话
3. **第三步**：（可选）添加 `search_history` 工具
4. **第四步**：测试，验证不同场景下的压缩效果

### 6. 讨论

这个方案更适合：
- ✅ 编程开发任务（轮次少，单轮内容长）
- ✅ 长周期迭代任务（主题会演进）
- ✅ 需要随时找回历史细节的场景

成本：
- 需要重新实现 `compress()` 方法的主要逻辑
- 但是结构化摘要、工具对配对清理这些现有逻辑都可以复用

---

## 当前状态

- ✅ 方案一（最小改动，protect_first_n=0）已完成
- ⏳ 方案二待开发

## 相关文件修改

方案一已经修改了：
1. `hermes_cli/config.py` — 添加 `protect_first_n: 0` 到 DEFAULT_CONFIG
2. `agent/context_engine.py` — 默认 `protect_first_n: int = 0`
3. `agent/context_compressor.py` — 构造函数默认 `protect_first_n=0`
4. `run_agent.py` — 构造调用时传 `protect_first_n=0`

要让配置生效，还需要在 `run_agent.py` 里从配置读取 `protect_first_n`：

```python
# 在读取 compression_protect_last 附近加上：
compression_protect_first = int(_compression_cfg.get("protect_first_n", 0))
```

然后传给 ContextCompressor。
