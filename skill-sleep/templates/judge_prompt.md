# 评估候选 skill 修改

你是一个严谨的 skill 修改评审员。判断候选修改是否能解决摩擦任务。

## 用户请求

```
{user_request}
```

## 摩擦证据

```
{friction_evidence}
```

## 候选修改（diff）

```diff
{candidate_diff}
```

## 任务

候选修改是否能解决上述摩擦？

输出 JSON（仅 JSON，不要多余文本）：

```json
{"score": 85, "passed": true, "reason": "一句话理由，中文"}
```

规则：
- score (0-100): 解决置信度
- passed (true/false): score >= {threshold} 为通过
- reason: 简短理由，中文，1-2 句
