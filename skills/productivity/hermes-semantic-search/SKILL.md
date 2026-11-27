---
name: hermes-semantic-search
description: 本地语义搜索 — 基于 ollama nomic-embed-text 的向量检索，索引知识库和会话。对标 OpenClaw MemorySearch 的混合检索管线（向量 + FTS5 + MMR + 时间衰减），纯本地运行无需外部 API。
version: 1.0.0
author: community
license: MIT
platforms: [linux, macos]
prerequisites:
  env_vars: []
  commands: [ollama]
metadata:
  hermes:
    tags: [semantic-search, vector, ollama, local, nomic, knowledge-base, session-search, retrieval]
    related_skills: [hermes-wiki, hermes-memory-providers, hermes-agent]
---

# Hermes 语义搜索

基于 ollama nomic-embed-text 的本地向量检索系统，对标 OpenClaw MemorySearch 的完整混合检索管线：向量语义搜索 + FTS5 全文匹配 → MMR 多样性重排 → 时间衰减排序。完全本地运行，不需要任何外部 API。

数据存储在 `~/.hermes/memory/vectors.db`，索引源包括知识库和对话转录。

## 架构

```
查询文本
  ├─→ ollama nomic-embed-text → 768维向量
  │     ├─→ vectors.db 余弦相似度（0.68 权重）
  │     └─→ state.db FTS5 BM25 文本匹配（0.32 权重）
  ├─→ 加权合并 → MMR 多样性重排（λ=0.7）
  └─→ 时间衰减（半衰期 30 天）→ Top-K 结果
```

### 数据源

- `source_type=wiki`: 知识库 `<!-- hermes:generated -->` 区块
- `source_type=session`: 用户+助手对话转录（排除 cron session）

### 参数表

| 参数 | 值 | 说明 |
|------|-----|------|
| 向量权重 | 0.68 | cosine similarity |
| 文本权重 | 0.32 | FTS5 BM25 rank |
| MMR λ | 0.7 | 相关性 vs 多样性平衡 |
| 时间半衰期 | 30天 | exp(-ln(2)*age/halflife) |
| 最低阈值 | 0.45 | 低于此分不返回 |
| chunk 大小 | 300字 | 50字重叠 |

## 依赖

```bash
# ollama + nomic-embed-text 模型（唯一外部依赖）
ollama pull nomic-embed-text
```

## 安装

安装本 skill 后，将脚本链接到 Hermes scripts 目录：

```bash
# 安装 skill（通过 hub）
hermes skills install hermes-semantic-search

# 链接脚本（skill 安装后执行）
ln -sf ~/.hermes/skills/productivity/hermes-semantic-search/scripts/embed.py ~/.hermes/scripts/embed.py
chmod +x ~/.hermes/scripts/embed.py
```

Hermes 启动时自动初始化 `vectors.db`（`init_db()` 幂等）。

## 命令

### 索引知识库

仅索引 `<!-- hermes:generated -->` 区块：

```bash
python3 ~/.hermes/scripts/embed.py index
python3 ~/.hermes/scripts/embed.py index --path 系统/踩坑记录.md
```

### 索引 Session 转录

索引最近 N 个对话（增量更新，内容未变的 chunk 跳过）：

```bash
python3 ~/.hermes/scripts/embed.py index-sessions
python3 ~/.hermes/scripts/embed.py index-sessions --limit 100
```

### 混合搜索

```bash
python3 ~/.hermes/scripts/embed.py search "查询文本"
python3 ~/.hermes/scripts/embed.py search "查询" --top-k 10
python3 ~/.hermes/scripts/embed.py search "查询" --pure-vector   # 仅向量
```

### 维护

```bash
python3 ~/.hermes/scripts/embed.py stats     # 统计
python3 ~/.hermes/scripts/embed.py cleanup   # 清理过期向量 + 陈旧检测
```

## 使用时机

- 用户搜索知识库/历史内容时，优先用混合搜索而非纯 FTS5 session_search
- 知识库文件更新后，运行 `index` 刷新向量
- 重要 session 结束后运行 `index-sessions` 增量索引对话转录
- 定期运行 `cleanup` 清理已删除文件的向量
- **建议 cron**：每天凌晨运行 `index-sessions --limit 20` 保持转录索引新鲜

## 搜索管线详解

### 第一步：向量搜索

将所有 chunk 的 nomic-embed-text embedding 与查询向量做余弦相似度，低于 0.45 阈值的直接丢弃。

### 第二步：FTS5 文本搜索

在 `state.db` 的 `messages_fts` 表上做 BM25 全文检索，将负 rank 值映射到 0-1 得分区间（`1/(1-rank)`）。FTS5 特殊字符（`-~()"^*`）自动清洗。

### 第三步：加权合并

- 向量结果：`score × 0.68`
- 文本结果：`bm25_score × 0.32`
- 同 key 命中双源：叠加得分
- 仅文本命中无向量：以 BM25 主导，用查询向量做 MMR fallback

### 第四步：MMR 多样性重排

Maximal Marginal Relevance 算法，在相关性和多样性之间平衡。λ=0.7 意味着每次选取时，70% 权重给混合得分，30% 权重惩罚与已选结果的相似度。

### 第五步：时间衰减

`final_score = hybrid_score × (0.8 + 0.2 × decay)`，其中 `decay = exp(-ln(2) × age_days / 30)`。30 天前半衰期，确保新鲜内容优先但不会完全淹没旧知识。

## 知识库索引策略

### 仅索引 generated 区块

知识库是人机双写的，只索引 Hermes 自己生成的内容（`<!-- hermes:generated:start -->` 到 `<!-- hermes:generated:end -->`），避免重复索引用户的原始内容或与知识库文件本身冲突。

### 增量更新

索引前先对比 `source + chunk_index` 的现有 content，内容未变则跳过 embed 调用，节省 ollama 计算。

## Session 索引策略

- 仅索引 `role IN ('user','assistant')` 消息
- 排除 cron session
- 最小消息数 > 2 才索引
- 拼接格式：`👤 ... 🤖 ...`
- 超大 session 截断至 30 chunks 防止 ollama OOM
- 重新索引时先 DELETE 旧 chunks 再 INSERT 新的

## 注意事项

- 首次索引 50 个 session 需要 3-5 分钟（取决于 ollama 速度 + 0.1s 限速）
- ollama 必须运行在 `localhost:11434`
- 知识库仅索引 `<!-- hermes:generated -->` 区块
- HTTP 500 自动重试 3 次（指数退避）
- 向量数据库文件 `~/.hermes/memory/vectors.db` 随索引增长，定期 `cleanup` 控制体积
