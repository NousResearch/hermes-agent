# model context alignment 重构方案

日期：2026-04-19
状态：方案已冻结（待实施）

## 背景
Hermes 当前在“模型上下文长度”上存在两条彼此相关、但实现分散的解析链路：
1. 运行时链路：`run_agent.py` 在构造 `AIAgent` 和执行 `switch_model()` 时，结合 `model.context_length` 与 `custom_providers[].models[*].context_length` 计算最终上下文长度，并传给 `agent.model_metadata.get_model_context_length()`。
2. 展示链路：`hermes_cli/model_switch.py` 负责 CLI 与 gateway 的 `/model` 结果构造，但它当前主要依赖 `model_info.context_window` 或无配置感知的回退探测，因此在自定义 provider / 自定义 base_url / `custom_providers` 覆盖存在时，展示值可能与运行时不一致。

本地只读验证已经确认，该问题在更新后的 `main` 上仍然存在：
- 当前配置模型：`glm-5`
- 当前配置 provider：`infini-ai`
- 运行时解析出的上下文长度：`200000`
- `switch_model()` 返回的 `model_info.context_window`：`None`

这说明当前系统仍缺少一个可复用、可测试、面向“配置覆盖 + provider/runtime + 展示输出”统一口径的上下文长度解析抽象。

## 目标
1. 建立单一、可复用的上下文长度配置解析模块，统一 `model.context_length` 与 `custom_providers` 覆盖解析逻辑。
2. 让运行时与展示侧共享同一套“有效配置上下文长度”决策规则，消除重复实现和分叉行为。
3. 让 `/model` 展示在 models.dev 元数据缺失时，仍能稳定展示与运行时一致的上下文长度。
4. 保持改动范围最小，不干扰无关功能，不引入新的配置写入语义。
5. 为该行为补充明确的单元/回归测试，保证后续演进不再回归。

## 非目标
- 本轮不重做 `agent.model_metadata.get_model_context_length()` 的整体分层。
- 本轮不修改 models.dev 拉取、缓存、provider catalog 体系。
- 本轮不新增新的配置项。
- 本轮不改动用户的 `config.yaml` 结构。
- 本轮不尝试清理所有历史上下文长度相关调用点，只聚焦“运行时 + `/model` 展示”一致性主路径。

## 问题拆解

### 现状问题 1：配置覆盖解析逻辑重复
当前 `run_agent.py` 内部手动实现了：
- 全局 `model.context_length` 解析
- `custom_providers` 命中与 `context_length` 解析
- 非法值 warning 输出

而 `hermes_cli/model_switch.py` 没有复用这套逻辑，导致 CLI/gateway 的显示口径与运行口径不一致。

### 现状问题 2：展示侧只知道 models.dev，不知道“有效配置覆盖”
`switch_model()` 当前返回：
- `model_info`
- `capabilities`
- provider/base_url/api_mode 等

但它没有返回“显示用上下文长度”这一归一化字段。CLI 和 gateway 只能各自决定：
- 优先显示 `model_info.context_window`
- 否则再自行调用 `get_model_context_length()`

这既重复，也丢失了“配置覆盖优先于自动探测”的核心语义。

### 现状问题 3：行为分散，难以测试
目前要验证该逻辑，需要分别理解：
- `run_agent.py`
- `hermes_cli/model_switch.py`
- `gateway/run.py`
- `cli.py`

缺乏一个明确的、可单测的纯函数边界。

## 设计原则
1. **单一职责**：把“配置侧上下文长度解析”抽成独立模块，不继续埋在 `run_agent.py` 内。
2. **运行与展示解耦**：运行时继续消费“有效配置上下文长度”；展示侧消费“显示用上下文长度”，二者共享底层解析函数，但不强耦合界面格式。
3. **最小正确改动**：不大规模重排 `model_switch` 主流程，只在合适节点插入共享解析。
4. **可测试优先**：核心行为必须落在纯函数上，避免只能通过 CLI/gateway 大链路间接验证。
5. **向前兼容**：现有配置文件、现有 provider 命名、自定义 provider slug/base_url 匹配方式保持兼容。

## 目标架构

### 新增共享模块
建议新增模块：`agent/context_length_config.py`

职责限定为：
1. 解析正整数上下文长度值。
2. 统一解析 `agent_cfg` 中对某个 `(model, provider, base_url)` 生效的配置覆盖。
3. 提供展示侧的“显示用上下文长度”解析函数。

该模块不负责：
- 输出 CLI 文本
- 输出 gateway 文本
- 直接写配置
- 拉取 models.dev 网络数据

### 模块接口设计

#### 1. `_coerce_positive_context_length(raw)`
用途：
- 将 `int` / 数字字符串解析为正整数
- 非法值返回 `None`

#### 2. `resolve_config_context_length(model, provider, base_url, agent_cfg, warn=None)`
用途：
- 返回对当前模型/运行时生效的配置覆盖上下文长度

解析优先级：
1. 命中的 `custom_providers[].models[model].context_length`
2. 顶层 `model.context_length`
3. `None`

说明：
- `warn` 为可选回调，供运行时输出 warning；展示侧调用时可不传。
- custom provider 匹配既支持 `custom:<slug>` provider 名，也支持 base_url 命中。

#### 3. `resolve_display_context_length(model, provider, base_url, api_key, model_info, agent_cfg)`
用途：
- 返回 `/model` 应显示的上下文长度

解析优先级：
1. `resolve_config_context_length(...)`
2. `model_info.context_window`
3. `get_model_context_length(..., config_context_length=None)` 的自动探测结果

说明：
- 该函数将展示规则显式化，避免 CLI/gateway 继续各自拼装。

## 实施方案

### Phase 1：抽离共享解析模块
改动点：
- 新增 `agent/context_length_config.py`
- 将 `run_agent.py` 内重复的配置上下文长度解析逻辑迁移到该模块

预期收益：
- 运行时逻辑职责清晰
- 后续 CLI/gateway 可直接复用

### Phase 2：让 `switch_model()` 返回显示用上下文长度
改动点：
- 在 `ModelSwitchResult` 中新增字段：`display_context_length: Optional[int]`
- 在 `switch_model()` 完成 provider/base_url/model 归一化后，调用共享模块计算该字段

预期收益：
- `/model` 的显示逻辑从“自己猜”变为“消费统一结果对象”

### Phase 3：CLI 和 gateway 只消费统一字段
改动点：
- `cli.py`：优先显示 `result.display_context_length`
- `gateway/run.py`：优先显示 `result.display_context_length`

预期收益：
- 展示逻辑不再重复实现同类决策
- 更容易扩展到未来的 TUI / API surface

### Phase 4：补测试
新增/更新测试覆盖：
1. 共享解析模块单测
2. `switch_model()` 在 custom provider 配置覆盖下返回 `display_context_length`
3. gateway `/model` 输出使用统一显示值
4. 运行时 `AIAgent` 仍保持原有 config override 行为

## 受影响文件（预期）
- 新增：`agent/context_length_config.py`
- 修改：`run_agent.py`
- 修改：`hermes_cli/model_switch.py`
- 修改：`cli.py`
- 修改：`gateway/run.py`
- 新增或修改测试：
  - `tests/agent/...` 或 `tests/hermes_cli/...`
  - `tests/gateway/test_model_command_custom_providers.py`
  - 保留 `tests/run_agent/test_switch_model_context.py`

## 风险与控制

### 风险 1：解析优先级变化导致行为漂移
控制：
- 明确冻结优先级：`custom_providers per-model > model.context_length > auto-detect`
- 用测试锁定该优先级

### 风险 2：展示侧与运行时又出现二次漂移
控制：
- 展示侧不再自行决定 config override，仅消费 `display_context_length`

### 风险 3：warning 行为被破坏
控制：
- warning 只在运行时路径触发
- 展示侧解析不输出 warning，避免 `/model` 噪音

### 风险 4：引入循环依赖
控制：
- 共享模块只依赖轻量配置读取/metadata 查询，不依赖 CLI/gateway 主类
- 不把该模块放入 `run_agent.py` 反向依赖链复杂的位置

## 测试方案
1. 运行共享模块的目标单测，验证：
   - 有效整数
   - 字符串整数
   - 非法值
   - custom provider 命中
   - base_url 命中
2. 运行 `tests/run_agent/test_switch_model_context.py`
3. 运行 `tests/run_agent/test_invalid_context_length_warning.py`
4. 运行 `tests/gateway/test_model_command_custom_providers.py`
5. 如有必要，增加 `tests/hermes_cli/test_model_switch_display_context.py`

## 验收标准
1. 在当前配置（`glm-5` + `infini-ai`）下，运行时与 `/model` 展示都能得到 `200000`。
2. CLI 与 gateway 不再各自手写同类显示决策。
3. 配置覆盖解析逻辑集中在共享模块中。
4. 目标测试全部通过。
5. 文档与进度日志完整，能够从任意断点继续。

## 实施前断点
- 已确认问题在更新后的 `main` 上仍然存在。
- 已完成方案冻结。
- 下一步：创建进度日志并开始 Phase 1 实施。
