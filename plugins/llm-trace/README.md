# llm-trace 插件

把 Hermes Agent 每一次 OpenAI 兼容 `chat.completions` 调用（请求 / 响应 / 流式 chunk / 错误）落到本地文件，供事后排障与回放。

零核心入侵——所有逻辑都在本插件目录内，通过 Hermes 的 `pre_api_request` hook 切分 turn，通过运行时包装 `openai.resources.chat.completions.{Completions, AsyncCompletions}.create` 捕获实际请求 / 响应。

## 覆盖范围

| 调用类型 | 是否捕获 | 标签（call_kind） |
|---|---|---|
| 主对话（`run_conversation` 主循环） | ✅ | `main` |
| 辅助调用 `compression`、`vision`、`title_generation`、`web_extract`、`mcp` 等（`agent/auxiliary_client.py::call_llm`） | ✅ | `aux:<task>` |
| 同步流式（`stream=True`） | ✅，逐 chunk | 同上 |
| 异步流式 | ✅，逐 chunk | 同上 |
| 非 OpenAI SDK 路径（直接 httpx） | ❌（如有需要再加补丁） | — |

## 启用

1. 编辑 `~/.hermes/config.yaml`：

   ```yaml
   plugins:
     enabled:
       - llm-trace
   ```

2. （可选）选择输出目录：

   ```bash
   export HERMES_LLM_TRACE_DIR="$HOME/dev/hermes-traces"   # 自定义
   # 不设置则默认写入 $HERMES_HOME/llm-traces/
   ```

3. （可选）临时关闭：`HERMES_LLM_TRACE=0`，插件保持加载但不写文件。

下次启动 `hermes` 即生效（启动日志会出现 `llm-trace: openai SDK patched`）。

## 输出结构

```
$HERMES_HOME/llm-traces/
└── sessions/
    └── <session_id>/
        ├── turn-0001-20260524-104233/
        │   ├── _turn.json                     # turn 元信息（user_message_preview 等）
        │   ├── call-1748061753-abc123-main/   # 主对话 LLM 调用
        │   │   ├── request.json               # 完整 kwargs（messages / tools / temperature ...）
        │   │   ├── chunks.ndjson              # 流式：逐 chunk 一行 JSON（仅 streaming）
        │   │   └── response.json              # 非流式响应 / 流式 EOS 汇总 / 错误
        │   ├── call-1748061754-def456-aux_compression/
        │   │   └── ...
        │   └── call-1748061755-ghi789-aux_vision/
        │       └── ...
        ├── turn-0002-...
        └── aux/                               # 没有 active turn 时的 fallback
            └── call-...
```

## chunk 文件格式（NDJSON）

每行一个 chunk：

```json
{"idx": 0, "ts": 1748061753.123, "chunk": {"id": "...", "choices": [{"delta": {"role": "assistant"}}]}}
{"idx": 1, "ts": 1748061753.156, "chunk": {"id": "...", "choices": [{"delta": {"content": "你好"}}]}}
```

可直接 `tail -f chunks.ndjson` 实时观察首 token 延迟、payload 大小、reasoning 字段是否到位。

## 设计取舍

- **运行时包装 OpenAI SDK**：而不是改 `chat_completion_helpers.py`。Hermes 上游频繁演进，直接改核心文件每次 `git pull` 都会冲突。包装 SDK 调用层只暴露稳定接口（`create()`），未来升级 SDK 也只需小修。
- **栈帧探测识别辅助任务**：`auxiliary_client.call_llm(task=...)` 的 `task` 参数无法穿过 OpenAI SDK 边界，所以在 patch 里通过 `sys._getframe()` 反查调用栈拿到 `task`。这层依赖 Hermes 的目录结构，但只读 frame.locals，对运行没影响。
- **代理类透明转发**：流式包装不修改 SDK 的 `Stream` 类，而是返回一个代理对象，所有未拦截的属性（`response`、`__enter__` 等）通过 `__getattr__` 转发到原 stream，避免破坏 Hermes 内部对 stream 的诊断访问（如 `stream.response` 抓 OpenRouter 缓存头）。
- **请求脱敏**：`request.json` 里只记 `kwargs`，不写 `api_key` / `Authorization`（这些都不在 kwargs 里，但保险起见在 recorder 做了显式过滤）。

## 关闭/卸载

- 临时关：`HERMES_LLM_TRACE=0 hermes` → 插件加载但所有 `create()` 调用走原始路径。
- 永久关：从 `plugins.enabled` 删除 `llm-trace` 即可。SDK 补丁仅在 `register()` 时安装，未启用时不会触发。

## 已知局限

1. 只覆盖 `openai.resources.chat.completions`，不覆盖直接走 httpx 的自定义 provider。如果某个 provider 不经过 OpenAI SDK，就抓不到。
2. `aux:unknown` 表示 patch 拦到了一次 OpenAI 调用，但栈里既没有 `pre_api_request` 标记，也没有 `auxiliary_client.call_llm` 帧——通常是某些直接构造 `OpenAI()` 客户端的边角路径（少见）。
3. NDJSON 是 append-only，turn 目录里磁盘占用会随对话积累；建议定期清理旧 `sessions/<id>/`。
