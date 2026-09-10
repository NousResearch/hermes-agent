# 任务：优化 agent skill

你是一个 skill 优化器。以下 friction 任务在真实会话中反复出现，当前 skill 文档未能有效解决。请给出有界编辑。

## 当前 skill 文档

```
{skill_content}
```

## 摩擦任务卡片

{tasks_summary}

## 之前被拒绝的编辑（上下文）

{rejected_context}

## 要求

- 只做有界编辑：追加 pitfalls、补充规则、调整措辞。不要重构整个 skill 结构。
- 新增行不超过 30 行（`+` 行计数）。
- 只修改这一个 SKILL.md 文件，不改其他文件。
- 输出必须是合法 unified diff（`--- a/SKILL.md` / `+++ b/SKILL.md` 头 + `@@` hunk），可被 `git apply` 直接应用。
- 在 diff 之前用 1-3 句中文简要说明本次优化聚焦什么（summary），并列出 1-3 个 focused_on 要点（每行一个 `focused_on: ...`）。

### 输出格式

先写说明区（纯文本），再写 diff 代码块：

```
summary: <一句话说明本次改动解决什么摩擦>
focused_on: <要点1>
focused_on: <要点2>

```diff
--- a/SKILL.md
+++ b/SKILL.md
@@ ... @@
 ...
```
```
