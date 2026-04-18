# model context alignment 重构进度日志

日期：2026-04-19
状态：进行中
关联方案：`docs/plans/2026-04-19-model-context-alignment-refactor-plan.md`

## 当前阶段
- Phase 0：问题确认与方案冻结
- Phase 1：准备进入共享解析模块抽离

## 基线状态
- Hermes 已完成 `hermes update`。
- `config.yaml` 已迁移到 `v19`。
- 当前分支为 `main`。
- 当前工作树包含历史未提交改动，实施时只处理与本任务直接相关的文件。
- 只读验证已确认问题仍存在：当前配置下运行时上下文长度为 `200000`，而 `switch_model()` 返回的 `model_info.context_window` 为 `None`。

## 本轮待办
- [x] 审阅相关代码路径与现有测试
- [x] 审阅 `docs/plans` / `docs/progress` 文档约定
- [x] 冻结实施方案并落盘
- [x] 抽离共享上下文长度配置解析模块
- [x] 让 `switch_model()` 返回统一显示值
- [x] 调整 CLI/gateway 消费统一显示值
- [ ] 补测试并验证

## 执行日志

### 2026-04-19 / Step 1
- 动作：复核本地 Hermes 升级后状态，确认当前分支、gateway 状态、配置版本与问题是否仍存在。
- 结果：确认 `hermes update` 与 `config migrate` 已完成，gateway 正常运行；问题仍存在，具备修复必要性。
- 影响文件：无。
- 备注：此步为只读验证。

### 2026-04-19 / Step 2
- 动作：只读验证当前默认模型 `glm-5` 在 `infini-ai` / custom runtime 下的运行时上下文长度与显示侧 metadata。
- 结果：
  - 运行时上下文长度：`200000`
  - `switch_model()` 成功，但 `model_info.context_window = None`
  - 证明显示侧与运行时侧仍存在口径分叉
- 影响文件：无。
- 备注：此步为是否需要修复的最终确认。

### 2026-04-19 / Step 3
- 动作：审阅 `run_agent.py`、`hermes_cli/model_switch.py`、`gateway/run.py`、`agent/model_metadata.py` 与相关测试，定位重复解析逻辑与可抽象边界。
- 结果：确认最合适的抽象边界是“配置覆盖上下文长度解析”和“显示用上下文长度解析”，应下沉为共享模块，避免继续把决策分散在 CLI/gateway/run_agent 中。
- 影响文件：无。
- 备注：方案要求优先模块化、解耦、可测试，此步用于冻结职责分层。

### 2026-04-19 / Step 4
- 动作：核对仓库内现有 `docs/plans` / `docs/progress` 风格，确认本轮方案与日志的组织方式。
- 结果：确认应沿用项目既有中文计划/进度格式，并将本轮文档分别落盘到 `docs/plans` 与 `docs/progress`。
- 影响文件：无。
- 备注：保证后续实施过程可追踪、可断点续作。

### 2026-04-19 / Step 5
- 动作：冻结本轮重构方案并落盘。
- 结果：已创建：
  - `docs/plans/2026-04-19-model-context-alignment-refactor-plan.md`
  - `docs/progress/2026-04-19-model-context-alignment-refactor-progress.md`
- 影响文件：
  - `docs/plans/2026-04-19-model-context-alignment-refactor-plan.md`
  - `docs/progress/2026-04-19-model-context-alignment-refactor-progress.md`
- 备注：在方案落盘前未改动业务代码。

### 2026-04-19 / Step 6
- 动作：新增共享模块 `agent/context_length_config.py`，并将 `run_agent.py`、`hermes_cli/model_switch.py`、`cli.py`、`gateway/run.py` 接入统一的配置上下文长度解析与显示上下文长度解析逻辑。
- 结果：
  - 配置覆盖解析从 `run_agent.py` 内联逻辑抽离为共享模块。
  - `switch_model()` 新增 `display_context_length` 字段，供 CLI 与 gateway 统一展示。
  - `AIAgent.switch_model()` 现在会在切模型后重新从配置解析有效上下文长度，不再只复用旧的全局值。
  - gateway `_format_session_info()` 也改为复用共享配置解析逻辑。
- 影响文件：
  - `agent/context_length_config.py`
  - `run_agent.py`
  - `hermes_cli/model_switch.py`
  - `cli.py`
  - `gateway/run.py`
- 备注：该步完成后，系统内主要上下文长度解析口径已统一，但仍需测试验证。

### 2026-04-19 / Step 7
- 动作：补充共享模块单测与 custom provider 显示值回归测试，并修正 CLI model picker state 透传完整 `agent_cfg`。
- 结果：新增/更新测试覆盖：
  - 配置上下文长度基础解析
  - custom provider 优先级
  - `/model` 返回统一 display_context_length
  - picker 路径的配置透传
- 影响文件：
  - `tests/agent/test_context_length_config.py`
  - `tests/hermes_cli/test_model_switch_custom_providers.py`
  - `cli.py`
- 备注：下一步执行目标测试集，验证本轮改动是否闭环。

### 2026-04-19 / Step 8
- 动作：执行目标测试集并做实际配置行为复核。
- 结果：
  - `scripts/run_tests.sh` 因环境无法安装 `pytest-split` 未能作为最终执行入口完成。
  - 改用激活 venv 后的 `python -m pytest` 直接运行本轮目标测试，共通过 28 个用例。
  - 实际配置复核结果：当前默认配置下 `RUNTIME_CTX=200000`，`DISPLAY_CTX=200000`，显示与运行时已对齐。
- 影响文件：无。
- 备注：本轮验证已覆盖功能正确性；`scripts/run_tests.sh` 的外部依赖问题属于环境限制，不是本次代码回归。

### 2026-04-19 / Step 9
- 动作：执行更大范围的定向回归，覆盖 `model_switch` 生态、gateway 模型切换持久化/覆盖链路、run_agent 上下文初始化与 CLI 低上下文告警面。
- 结果：
  - `tests/hermes_cli/test_model_switch_*.py`、`tests/hermes_cli/test_user_providers_model_switch.py`、`tests/hermes_cli/test_custom_provider_model_switch.py`：40 通过。
  - `tests/gateway/test_model_switch_persistence.py`、`tests/gateway/test_session_model_override_routing.py`、`tests/gateway/test_session_model_reset.py`、相关 `/model` / session info 测试：24 通过。
  - `tests/run_agent/test_plugin_context_engine_init.py`、`tests/run_agent/test_switch_model_context.py`、`tests/run_agent/test_invalid_context_length_warning.py`、`tests/cli/test_cli_context_warning.py`：19 通过。
  - 合计新增更大范围回归：83 通过，0 失败。
- 影响文件：无。
- 备注：当前回归信号良好，未发现本次重构引入的扩散性问题。

## 当前断点
- 已完成：问题确认、职责梳理、方案冻结、共享模块抽离、主流程接线、测试补充。
- 已完成：目标测试执行、最终行为复核、更大范围定向回归。
- 未完成：无。
- 下一步：如主人需要，可继续做更大范围回归或整理提交。
