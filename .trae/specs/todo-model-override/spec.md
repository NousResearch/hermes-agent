# Todo 任务模型覆盖功能规格

## Why

当前 `todo` 工具的每个任务只能指定 `id`、`content` 和 `status`，但用户希望在创建子任务时可以指定使用不同的模型和提供商。主要使用场景是**多模型交叉验证**：同一个任务使用不同具体模型并行分析，然后比较结果。

`delegate_task` 工具已支持通过 `delegation.provider` 配置子代理模型，但无法为不同的子任务指定不同的模型。需要扩展 `todo` 工具的 schema，让每个任务项可以声明 `model` 和 `provider` 字段，并在执行 `delegate_task` 时透传给子代理。

## What Changes

- 扩展 `tools/todo_tool.py` 中 `TODO_SCHEMA` 的任务项 schema，新增可选的 `model` 和 `provider` 字段
- 在 `delegate_task` 的任务项 schema 和参数处理中支持 `model` 和 `provider` 字段，按任务级别覆盖全局配置
- 优先级：任务级 provider + model > delegation.model/provider > 继承父代理

## Impact

- Affected specs: `delegate_task` schema (tasks 数组项新增 model/provider 字段)
- Affected code:
  - `tools/todo_tool.py` — TODO_SCHEMA 扩展
  - `tools/delegate_tool.py` — per-task model/provider override 逻辑

## ADDED Requirements

### Requirement: Todo 任务项支持 model 和 provider 字段

系统应在 todo 任务的 schema 中支持可选的 `model` 和 `provider` 字段，允许用户为每个任务指定不同的模型和提供商。

#### Scenario: 创建带 model 和 provider 的任务

- **WHEN** 用户调用 `todo([{"id": "task-1", "content": "分析代码", "status": "pending", "provider": "openrouter", "model": "google/gemini-2.5-flash"}])`
- **THEN** 任务项被正确存储，provider 和 model 字段被记录
- **AND** 后续 `delegate_task` 调用该任务时使用指定的 provider 和 model

### Requirement: 任务级模型覆盖

`delegate_task` 的任务项应支持 `model` 和 `provider` 字段，优先级高于全局 `delegation.model` 和 `delegation.provider`。

#### Scenario: 批量任务使用不同模型

- **WHEN** 用户调用 `delegate_task` 并传入多个任务，每个任务指定不同的 provider/model
- **THEN** 每个子代理使用其对应的模型运行
- **AND** 未指定 model/provider 的任务回退到全局配置

### Requirement: 模型格式解析

系统应支持两种 model 格式的解析：

1. **分离格式**: `provider: "openrouter", model: "gemini-2.5-flash"` — 直接在任务项中指定
2. **纯模型名**: `"model-slug"` — 使用任务级 provider 或全局 provider

#### Scenario: 使用模型格式

- **WHEN** 用户在 tasks 中指定:
  - `"provider": "openrouter", "model": "claude-3-sonnet"` (分离格式)
  - `"model": "gpt-4o"` (纯模型名，配合 provider)
- **THEN** 系统正确解析每种格式并创建对应的子代理

## MODIFIED Requirements

### Requirement: delegate_task tasks schema

**修改**: 在 `delegate_task` 的 tasks 数组项中新增 `model` 和 `provider` 属性，允许为每个子任务指定模型和提供商。

旧 schema:
```json
{"goal": "...", "context": "...", "toolsets": [...], "role": "leaf"}
```

新 schema:
```json
{"goal": "...", "context": "...", "toolsets": [...], "role": "leaf", "model": "claude-3-sonnet", "provider": "openrouter"}
```

## REMOVED Requirements

无。

## 优先级链

```
任务级 provider + model 字段
    ↓ (如果为空)
delegation.provider + delegation.model (全局配置)
    ↓ (如果为空)
继承父代理的 provider 和 model
```

## 使用示例

```python
# 多模型交叉验证：同一任务使用不同具体模型并行分析
todo([
    {"id": "analysis-1", "content": "分析这段代码的安全漏洞", "status": "pending", "provider": "openrouter", "model": "google/gemini-2.5-flash"},
    {"id": "analysis-2", "content": "分析这段代码的安全漏洞", "status": "pending", "provider": "openrouter", "model": "anthropic/claude-3-5-sonnet"},
])

# 方式 1: 分离格式 (provider + model 分开)
todo([
    {"id": "task-1", "content": "使用 Gemini Flash", "status": "pending", "provider": "openrouter", "model": "google/gemini-2.5-flash"},
    {"id": "task-2", "content": "使用 Claude Sonnet", "status": "pending", "provider": "openrouter", "model": "anthropic/claude-3-5-sonnet"},
])

# 方式 2: 纯模型名（配合 provider 字段）
todo([
    {"id": "task-1", "content": "只用模型名", "status": "pending", "provider": "openai", "model": "gpt-4o"},
])

# delegate_task 直接使用
delegate_task(tasks=[
    {"goal": "分析这段代码", "provider": "openrouter", "model": "google/gemini-2.5-flash"},
    {"goal": "分析这段代码", "provider": "openrouter", "model": "anthropic/claude-3-5-sonnet"},
])
```
