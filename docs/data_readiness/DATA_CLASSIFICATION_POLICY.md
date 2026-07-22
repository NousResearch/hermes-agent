# DATA_CLASSIFICATION_POLICY

## Purpose
Provide a single enforceable classification hierarchy for all data under Hermes/BlackGold control, with an explicit restoration freeze on live confidential data.

## Classification tiers

| Tier | Label | Examples | Handling |
|------|-------|----------|----------|
| T0 | Public |公开发布资料, 匿名化统计 | 可公开访问 |
| T1 | Internal | Hermes配置、日志模式 | 最小访问控制 |
| T2 | Confidential | BlackGold操作元数据、泛化deal资料 | 仅授权代理与服务 |
| T3 | Restricted | 识别个人/投资者名单、PII | 人工门审批+加密+审计日志 |
| T4 | Secret | 凭证、密钥、APi令牌 | 永不允许未授权写入; 仅人工可访问 |

## Restoration freeze
生产恢复严格冻结：
- 未批准的法定PII、金融、银行、投资者、员工、教会成员、客户或个人信息不得被摄取。
- 允许摄取：仅合成/模拟数据、授权测试夹具、明确公开参考。
- 突破冻结必须由Arthur Lyons书面批准。

## Marking规则
- 所有新文件必须在文件名或YAML frontmatter中运行 tier 标记。
- 群集: `tier:<T0..T4>` + `state: synthetic|live|quarantine`

## 处理规则
- T4必须在存储前脱敏; 必须单独加密。
- 不可逆地将T2/T3/T4数据转移到公有存储。
- 任何代理不得绕过quarantine分类递送路径。
