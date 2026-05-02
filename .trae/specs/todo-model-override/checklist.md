# Checklist

- [x] TodoStore 正确存储和返回包含 model/provider 字段的任务项
- [x] TODO_SCHEMA schema 定义包含可选的 model 和 provider 字段
- [x] delegate_task tasks schema 支持 model 和 provider 字段
- [x] 任务级 model/provider 字段正确传递给 _build_child_agent
- [x] 纯模型名格式正确解析（使用任务级或全局 provider）
- [x] 分离格式 provider + model 字段正确处理
- [x] 优先级链正确工作：task provider/model > delegation > parent
- [x] 单元测试覆盖 model/provider 字段存储和解析逻辑
- [x] 不影响现有 todo 和 delegate_task 功能
- [x] 多模型交叉验证场景验证（同一任务不同模型并行）
- [x] delegate_task 的 batch mode 正确使用 per-task model 配置
