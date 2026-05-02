# Tasks

- [x] Task 1: 扩展 todo_tool.py 的 TODO_SCHEMA，添加 model 和 provider 字段支持
  - 在 TODO_SCHEMA 的 tasks.items.properties 中添加可选的 `model` 和 `provider` 字段
  - 更新 `TodoStore.write()` 和 `TodoItem` 数据结构以支持 model 和 provider 字段的存储和传递

- [x] Task 2: 扩展 delegate_task schema 支持任务级 model 和 provider 覆盖
  - 在 DELEGATE_TASK_SCHEMA 的 tasks.items.properties 中添加 `model` 和 `provider` 字段
  - 在 `delegate_task()` 函数中处理 per-task model/provider 参数

- [x] Task 3: 实现模型引用解析逻辑
  - 创建 `_resolve_task_model()` 函数处理两种格式的解析:
    - 分离格式: `provider` 和 `model` 字段直接使用
    - 纯模型名: 使用任务级 provider

- [x] Task 4: 集成 model 解析到子代理构建流程
  - 修改 `_build_child_agent()` 接收 per-task 的 model/provider
  - 修改 `delegate_task()` 在构建子代理时传递解析后的凭证

- [x] Task 5: 单元测试验证
  - 编写测试验证 todo tool 的 model/provider 字段存储
  - 编写测试验证两种 model 格式的解析
  - 验证 delegate_task batch mode 正确使用 per-task model 配置

# Task Dependencies

- [Task 2] 依赖 [Task 1] — delegate_task schema 需要与 todo schema 保持一致
- [Task 3] 依赖 [Task 1] — 解析逻辑需要理解 schema 结构
- [Task 4] 依赖 [Task 2] 和 [Task 3] — 集成需要 schema 和解析逻辑都就绪
