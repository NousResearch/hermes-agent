# DATA_ACCESS_MATRIX

## Roles
- Marcus - Commander
- Grant - Private Equity
- Malcolm - Finance
- Julian - Banking/GCI
- Elijah - Legal/Contract
- Jordan - Sales/Lead
- Miles - Coding
- Naomi - Security
- Olivia - Operations
- Grace - Church Tech
- Caleb - Industrial
- David - Strategy
- Sophia - UI/Brand
- Victor - Bug Catcher/System Monitor
- System Services - gateway, cron, backup

## Permission classes
- `none`
- `read`
- `append`
- `admin`

## Synthetic-only constraint
Until Arthur expressly authorizes live ingestion, all授权读取和写入权限适用于合成数据模块和测试数据。

## 矩阵

| Resource | Marcus | Grant | Malcolm | Julian | Elijah | Jordan | Miles | Naomi | Olivia | Grace | Caleb | David | Sophia | Victor | System Services |
|----------|--------|-------|---------|--------|--------|--------|-------|-------|--------|-------|-------|-------|--------|--------|-------|
| 系统记录 (deals/tasks/documents/contacts) | admin | read | read | append | read | read | admin | read | append | none | read | read | read | read | append |
| 文档保险库 (staging/validated/archive) | admin | read | read | append | read | read | read | read | append | none | read | read | read | read | append |
| 报告/分析 | read | read | read | read | read | read | read | read | read | read | read | read | read | read | none |
| 审计日志 | read | none | none | none | none | none | read | admin | read | none | none | none | read | read | append |
| 配置 | read | none | read | none | none | none | read | admin | none | none | none | read | read | read | none |
| 备份/恢复 | read | none | none | none | none | none | read | admin | read | none | none | none | read | read | admin |

## 例外
- 人工门旁路: 仅Arthur Lyons可批准临时提升权限。
- 写访问需单独审批。
