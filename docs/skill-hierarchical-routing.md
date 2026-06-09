---
{"tags":["reference","config","skills"],"created":"2026-06-09"}
---

# Skill Hierarchical Routing (L1/L2/L3)

> Added in PR #??? — replaces the full ``<available_skills>`` block with a compact L1 guide when enabled.

## Configuration

Add these keys to ``config.yaml``:

```yaml
skills:
  # "compact" (default) — full index in system prompt
  # "l1_l2" — compact L1 guide, full index on-demand
  routing_mode: l1_l2

  # Skills always available in the system prompt (L1)
  # Only the most frequently used skills should be listed here.
  l1_skills:
    - [cron-management, "定时任务管理"]
    - [context-optimization, "上下文优化"]
    - [feishu-compression, "飞书消息压缩"]
    - [hermes-agent, "核心配置"]
    - [memory-management, "记忆管理"]
```

## How it works

| Mode | Prompt size | Latency | Behavior |
|------|-------------|---------|----------|
| ``compact`` (default) | ~13,670 tokens | 0ms | Full index in prompt, LLM matches by keyword |
| ``l1_l2`` | ~100 tokens | 0ms (L1) / ~500ms (L2) | L1 guide in prompt, full index on-demand via ``skills_index.yaml`` |

## Architecture

```
                System Prompt (~100 tokens, L1)
                ┌──────────────────────┐
                │  5高频skill关键词    │
                └──────────────────────┘
                          │
                  请求匹配L1?
                  ┌───────┴───────┐
                 YES              NO
                  │               │
            [直接执行]      调 read_file()
                            L2 映射表 (~5,000 chars)
                                    │
                              按分类/描述选skill
                                    │
                              调 skill_view(name)
                            L3 SKILL.md全文
                                    │
                              按步骤执行
```

## Migration

1. Set ``skills.routing_mode: l1_l2`` in ``config.yaml``
2. Populate ``skills.l1_skills`` with your top 5-7 skills
3. (Optional) Create ``~/.hermes/skills_index.yaml`` with the full catalog
4. Restart Hermes — the next session uses the compact mode

To revert, set ``routing_mode: compact`` or remove the ``skills`` block.
