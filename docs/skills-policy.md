# Skills Policy

Updated: 2026-04-18

## Purpose

这份文档定义 Hermes 什么时候应该创建、修补、删除或忽略 skill。

核心原则：**skill 是长期可复用流程，不是任务日志。**

## Create a Skill When

满足任一条件就应考虑创建：
- 完成了一个多步骤复杂任务（通常 5+ 工具调用）
- 解决了一个以后大概率会再次出现的问题
- 用户对做法给出过关键纠正
- 发现了稳定的环境特性、依赖顺序、验证套路
- 某个工作流值得标准化，减少下次重复思考

## Patch a Skill When

出现以下情况应立即 patch：
- 命令已过时
- 路径/配置已变化
- 文档缺少关键前置条件
- 文档遗漏了常见坑
- 文档没有说明如何验证结果
- 技能与用户当前偏好冲突

## Do Not Create Skills For

以下内容不应做成 skill：
- 一次性聊天结论
- 单次任务进度
- 临时排查记录
- 太短且毫无复用价值的操作
- 用户的短期提醒事项

## Required Skill Structure

一个合格的 skill 至少应包含：
- Trigger / 何时使用
- Preconditions / 前置条件
- Steps / 标准步骤
- Verification / 验证方式
- Pitfalls / 常见坑

## Quality Bar

写 skill 时遵守：
- 给出准确命令、路径、工具名
- 不写空话，比如“检查一下”“适当调整”
- 尽量说明失败时如何判断
- 优先记录已验证过的方法
- 如果有 OS/平台差异，要写清楚

## Maintenance Loop

建议结合 cron 定期检查：
- 哪些近期任务值得沉淀为 skill
- 哪些 skill 最近执行时出现问题
- 哪些 skill 需要 patch 或删除

## Delete or Retire a Skill When

满足任一条件可删除或退休：
- 依赖的工具已不存在
- 方案已经被新流程完全替代
- 长期没人使用且容易误导
- 多次验证都发现说明已不可靠

## Rule of Thumb

如果这条经验能让未来的 Hermes 少问一次、少踩一个坑、少重复一次排查，它就值得进入 skill 系统。
