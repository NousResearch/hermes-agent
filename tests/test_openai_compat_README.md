# OpenAI API 兼容性测试套件

验证自定义 API 端点是否符合 OpenAI Chat Completions 接口规范。

## 测试覆盖

| 测试类 | 测试数 | 验证内容 |
|--------|--------|----------|
| TestBasicConnectivity | 2 | 端点可达性、/models 端点 |
| TestChatCompletions | 7 | id/object/created/model/choices/usage 字段 |
| TestChoiceStructure | 5 | index/message/role/content/finish_reason |
| TestStreamingResponse | 6 | 流式迭代器、delta 格式 |
| TestParameterCompatibility | 6 | temperature/max_tokens/top_p/stop/extra_body/多轮对话 |
| TestErrorHandling | 3 | 无效模型/空消息/无效 role |
| TestResponseTiming | 1 | 响应延迟 <30s |

## 环境变量配置

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TEST_OPENAI_BASE_URL` | `https://ai-pool.evebattery.com/v1` | API 基础 URL |
| `TEST_OPENAI_API_KEY` | (必填) | API 密钥 |
| `TEST_OPENAI_MODEL` | `Qwen3-235B-A22B-w8a8` | 模型名称 |
| `TEST_OPENAI_SSL_VERIFY` | `true` | 设为 `false` 禁用 SSL 验证 |
| `TEST_OPENAI_NO_PROXY` | `false` | 设为 `true` 绕过系统代理 |
| `TEST_OPENAI_TIMEOUT` | `60` | 请求超时秒数 |

## 运行测试

```powershell
# Windows PowerShell
$env:TEST_OPENAI_API_KEY="your-api-key"
$env:TEST_OPENAI_SSL_VERIFY="false"
$env:TEST_OPENAI_NO_PROXY="true"
.\.venv\Scripts\python.exe -m pytest tests/test_openai_compat.py -v -o "addopts="

# Linux/macOS
export TEST_OPENAI_API_KEY="your-api-key"
export TEST_OPENAI_SSL_VERIFY="false"
export TEST_OPENAI_NO_PROXY="true"
python -m pytest tests/test_openai_compat.py -v -o "addopts="
```

## 测试结果示例

```
tests/test_openai_compat.py::TestBasicConnectivity::test_endpoint_reachable PASSED
tests/test_openai_compat.py::TestChatCompletions::test_has_id_field PASSED
tests/test_openai_compat.py::TestStreamingResponse::test_stream_returns_chunks PASSED
tests/test_openai_compat.py::TestParameterCompatibility::test_multiple_messages PASSED
...
======================== 30 passed in 19.55s ========================
```
