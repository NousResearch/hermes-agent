# Hermes Self Evolution Healthcheck

## 结论
新增独立健康检查脚本，用一条命令快速判断自主进化链路是否健康或发生漂移。

## 文件
- Script:
  - `/Users/blank/.hermes/scripts/hermes_self_evolution_healthcheck.py`

## 检查内容
脚本会输出结构化 JSON，并检查：
- 核心文件是否存在
- report 与 next_actions 的 top priority 是否对齐
- snapshot alignment 标记是否为 true
- latest/history snapshot 路径是否对齐
- rule engine version 是否在 next_actions / snapshot / status doc 中可见
- executed_low_risk_action 是否存在

## 运行方式
```bash
python3 /Users/blank/.hermes/scripts/hermes_self_evolution_healthcheck.py
```

## 返回语义
- `status: healthy`
  - 说明当前链路关键产物存在且主对齐项通过
- `status: drift-detected`
  - 说明至少一个关键检查失败，应根据 `checks[].name` 和 `checks[].detail` 定位

## 当前已验证状态
最近一次手动运行结果：
- `status: healthy`
- `top_priority: hermes-self-evolution-maintenance`
- `rule_engine_version: state-based-v1.4`

## 已接入 cron
- `b9534290f048` Hermes 自主进化健康检查
  - schedule: `every 2880m`
  - deliver: `origin`

## 设计原则
- 不直接修改系统，只做检测
- 输出结构化 JSON，方便 cron / 人工 / 后续脚本消费
- 优先检查“是否漂移”，而不是输出大段描述
