# Hermes → Open WebUI 性能分析: SSE Batching + Filter Function

## 问题

Hermes 接入 Open WebUI (Responses API 模式) 时，长流式输出导致浏览器严重卡顿。
典型症状: UI 冻结、滚动跳跃、spinner 不停止、CPU 100%。

## 根因分析 (三层叠加)

| 层级 | 根因 | 影响 |
|------|------|------|
| **Hermes API Server** | 每 token 发送一个 SSE 事件 (~500 事件/20s) | Open WebUI 每次重渲染 markdown |
| **Open WebUI 前端** | Markdown.svelte 每次 token 变化完整重解析 + Katex | 每帧 600ms (Safari v0.7.2 实测) |
| **Tool Call DOM 膨胀** | write_file 24KB raw JSON 塞进 tool card <pre> | 数百额外 DOM 节点，无虚拟滚动 |

## 修复 1: SSE Token Batching (Hermes 服务端)

**文件:** gateway/platforms/api_server.py → _write_sse_responses()

**改动:** _dispatch() 函数增加 50ms token 缓冲，积累文本 delta 后批量 emit:

```
# Before: 每个 token 立即 emit
elif isinstance(it, str):
    await _emit_text_delta(it)

# After: 缓冲 50ms 后批量 emit
elif isinstance(it, str):
    _batch_buf.append(it)
    if _batch_timer is None:
        _batch_timer = asyncio.create_task(_batch_flush_after(0.05))
```

**附带修复:**
- nonlocal _batch_timer — 修复 UnboundLocalError
- response.completed 内容裁剪 — 848KB+ → ~8KB，解决 SSE hang
- client_max_size 10MB — 修复 413 Request Body Too Large
- Catch-all 异常处理器 — 修复 TransferEncodingError

**量化:**

| 指标 | 修复前 | 修复后 | 改善 |
|------|--------|--------|------|
| SSE 事件数 (20s 响应) | ~500 | ~20 | **-96%** |
| Open WebUI re-render 次数 | 500 次 | 20 次 | **-96%** |
| response.completed 大小 | 848 KB | ~8 KB | **-99%** |
| TransferEncodingError | 偶发 | 0 | **消除** |

---

## 修复 2: Open WebUI Filter Function (前端)

**文件:** references/filter-function-v3.py

**原理:** 在 Open WebUI 的 SSE 中间件层拦截 Hermes 的事件，到达前端渲染器之前裁剪大字段。

**处理链路:**

```
Hermes SSE  -> Open WebUI middleware -> FILTER 拦截 -> 前端渲染
              |                                      
     response.output_item.added  -> function_call      -> 美化参数 + 记录 call_id->name
                                 -> function_call_output -> 裁剪输出
     response.output_item.done   -> function_call_output -> 查找 name + 裁剪/美化
     response.completed          -> 全量裁剪 (content, query, pattern, code...)
```

**功能特性:**

| 功能 | 说明 |
|------|------|
| **Emitter 美化** | 15 个 tool 的 emoji 摘要 (path (24.5 KB) 代替 raw JSON) |
| **输出摘要** | JSON 输出转一行文案 (5 results 代替 {"data":{"web":[...]}}) |
| **call_id->name 追踪** | 精确映射 tool name (修复旧版 call_id.split("_")[0] 的猜错问题) |
| **多 part 输出** | 遍历全部 output parts (修复旧版只取 output[0] 的丢失问题) |
| **response.completed 裁剪** | 848KB -> ~8KB (最大单次性能收益) |
| **inline-output hint** | 在用户消息中注入提示，让 Hermes 优先内联输出 |

**量化:**

| 指标 | 无 Filter | 有 Filter | 改善 |
|------|-----------|-----------|------|
| Tool card DOM 节点 (write_file 24KB) | ~300+ nodes | ~5 nodes | **-98%** |
| response.completed SSE 大小 | 848 KB | ~8 KB | **-99%** |
| 首次渲染耗时 | ~600ms | ~80ms | **-87%** |
| Katex CPU 占用 | 117ms/帧 | ~30ms/帧 | **-74%** |

---

## 综合效果

| 指标 | 初始状态 | 仅 Batching | Batching + Filter | 总改善 |
|------|----------|-------------|-------------------|--------|
| SSE 事件/响应 | 500 | 20 | 20 | **-96%** |
| DOM 节点 (tool card) | 300+ | 300+ | 5 | **-98%** |
| CPU/帧 | 600ms | 160ms | ~80ms | **-87%** |
| response.completed | 848KB | 848KB | ~8KB | **-99%** |
| UI 冻结 | 是 | 轻微 | 无 | **解决** |

---

## 部署指南

### Hermes 服务端
合并 PR 后 git pull && hermes gateway restart

### Open WebUI Filter
1. 复制 references/filter-function-v3.py 到 Open WebUI 的 data/functions/ 目录
2. 重启 Open WebUI (自动发现)
3. Admin Settings -> Functions -> 激活 "Hermes Tool Sanitizer" -> 设为 Global

或者通过 API:
```
# 1. 获取 token
TOKEN=$(curl -s http://127.0.0.1:7899/api/v1/auths/signin \
  -H "Content-Type: application/json" \
  -d '{"email":"...","password":"..."}' | jq -r .token)

# 2. 创建 filter
curl -s http://127.0.0.1:7899/api/v1/functions/create \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"id":"hermes-tool-sanitizer","name":"Hermes Tool Sanitizer","type":"filter","content":"...","is_active":true,"is_global":true}'
```

## 相关 Issue/PR

| 链接 | 说明 |
|------|------|
| [hermes-agent#17537](https://github.com/NousResearch/hermes-agent/issues/17537) | Feature Request: SSE batching |
| [hermes-agent#17541](https://github.com/NousResearch/hermes-agent/pull/17541) | PR: SSE batching + trimming |
| [open-webui#20878](https://github.com/open-webui/open-webui/discussions/20878) | UI freezes during streaming (已解决于 v0.8.9) |
| [open-webui#21884](https://github.com/open-webui/open-webui/pull/21884) | PR: fast-path JSON.stringify skip (已合并) |
