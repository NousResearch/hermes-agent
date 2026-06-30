# 本地 demo：用同样的接入方式跑通 hermes

> **面向**：技术（产品演示支撑）
> **目标**：在本机用 **`api_server` + Chat Completions + SSE 单向通道** 这套**与生产完全相同**的接入方式，搭一套最轻量的 hermes demo，用于向产品演示：
> 1. 这套接入方式能做什么（流式输出、工具调用）；
> 2. 这套接入方式**做不到**什么（AskUserQuestion 卡片）。
>
> **Provider**：OpenAI 兼容端点（你填自己的 key）。
> **客户端语言**：TypeScript / Node（Demo A 用官方 `openai` SDK，Demo B 零依赖）。

---

## 0. 架构

```
┌─────────────────────────────┐       ┌──────────────────────────────┐
│  demo 客户端（TypeScript）    │       │  hermes 服务端（Python，单进程）│
│  - demo_a: chat completion  │  HTTP │  hermes gateway run            │
│  - demo_b: /v1/runs (SSE)   │ ────► │  └─ api_server platform :8642  │
└─────────────────────────────┘  SSE  │       └─ AIAgent ──► 上游 LLM  │
                                  ◄─── └──────────────────────────────┘
```

- **接入方式与生产一致**：都是 OpenAI 兼容 Chat Completions + SSE。demo 跑通的现象 = 生产链路的现象。
- **语言分工**：hermes **服务端是 Python**，必须用 Python 安装（见步骤 2）；**demo 客户端用 TS**，仅消费 HTTP+SSE，语言无关。
- **轻量**：单进程、单端口；Demo A 一个 npm 依赖（`openai`），Demo B 零依赖。

---

## 1. 前置条件

| 项 | 要求 | 用于 |
|----|------|------|
| Python | 3.10+ | 安装运行 hermes 服务端 |
| Node | 20+（本机已 v22.15 ✅） | 跑 demo 客户端 |
| LLM Key | 任一 OpenAI 兼容端点的 API Key | 上游模型 |
| 端口 | 8642 空闲（或自定义） | api_server 监听 |

---

## 2. 安装 hermes（Python，服务端）

```bash
cd /Users/liepin/Documents/study/hermes-agent

# 虚拟环境隔离
python3 -m venv .venv-demo
source .venv-demo/bin/activate

# 安装（二选一）
uv pip install -e ".[all,dev]"     # 推荐，快；没有 uv 就 pip install uv
# 或
pip install -e ".[all,dev]"
```

验证：

```bash
hermes --version
```

---

## 3. 配置 Provider（OpenAI 兼容）

### 方式 A：交互式引导（推荐，最稳）

```bash
hermes
```

按提示选择：
- **Provider** → 选 `custom`（任意 OpenAI 兼容端点）
- **base_url** → 填你的端点，如 `https://api.openai.com/v1` 或公司内部端点
- **api_key** → 填你的 key
- **model** → 填模型名，如 `gpt-4o`、`gpt-4.1` 或公司内部模型

配置写入 hermes profile 目录，后续 `hermes gateway run` 自动复用。

### 方式 B：环境变量（适合脚本化）

在项目根目录建 `.env`：

```dotenv
# === 上游 LLM（OpenAI 兼容）===
OPENAI_API_KEY=sk-你的key
OPENAI_BASE_URL=https://api.openai.com/v1   # 替换为你的端点
# HERMES_MODEL=gpt-4o                        # 按端点支持的模型填
```

> hermes 的 Provider/Model 配置较丰富（多 provider、fallback、别名）。demo 用最小配置即可；完整选项见 `cli-config.yaml.example`。**方式 A 引导一次最省心。**

---

## 4. 启用 api_server

`api_server` 通过环境变量开关启用（源码：`gateway/config.py:1534-1558`）。在 `.env` 追加，或直接 export：

```bash
export API_SERVER_ENABLED=true          # 启用 api_server platform
export API_SERVER_KEY=demo-key-123      # 鉴权 Bearer token（自定义，客户端要带）
export API_SERVER_PORT=8642             # 端口，默认 8642
export API_SERVER_HOST=127.0.0.1        # 只监听本机；对外演示改 0.0.0.0
export API_SERVER_CORS_ORIGINS=*        # 浏览器演示需要 CORS；生产收紧
```

| 变量 | 作用 | demo 取值 |
|------|------|-----------|
| `API_SERVER_ENABLED` | 总开关 | `true` |
| `API_SERVER_KEY` | Bearer 鉴权 token（设了也会自动启用） | `demo-key-123` |
| `API_SERVER_PORT` | 监听端口 | `8642` |
| `API_SERVER_HOST` | 监听地址 | `127.0.0.1` |
| `API_SERVER_CORS_ORIGINS` | CORS 白名单 | `*`（演示用） |

---

## 5. 启动

```bash
hermes gateway run
```

看到 api_server 监听 8642 的日志即成功（前台运行，演示时保持开启）。

---

## 6. 验证（curl）

```bash
# 能力清单
curl -s http://127.0.0.1:8642/v1/capabilities \
  -H "Authorization: Bearer demo-key-123" | python3 -m json.tool

# 可用模型（hermes 把自己暴露为 "hermes-agent" 这个 model）
curl -s http://127.0.0.1:8642/v1/models \
  -H "Authorization: Bearer demo-key-123" | python3 -m json.tool
```

> **关于 `model` 字段**：api_server 把整个 hermes agent 包装成一个名为 `hermes-agent` 的 model（源码：`api_server.py:860-871`）。客户端请求里的 `model` 填 `hermes-agent` 即可，真正的上游模型由 hermes 的 Provider 配置决定。

---

## 7. 接入 Demo 代码（TypeScript）

> 完整可运行文件已落盘：`demo/demo_a.ts`、`demo/demo_b.ts`。两个 demo 都用「与生产一致」的接入方式。

### 7.0 准备

```bash
cd docs/ask-user-question-analysis/demo
npm init -y                 # 生成 package.json
npm pkg set type=module     # 必需：top-level await 需 ESM，不加 tsx 会报 cjs 错误
npm i openai                # Demo A 需要；Demo B 不需要
npm i -D tsx                # 运行 TS 的最简方式（也可零依赖原生跑，见 7.3）
```

### 7.1 Demo A：Chat Completions 流式（openai SDK，最简）

`demo_a.ts` —— hermes 是 OpenAI 兼容，直接套官方 SDK，SSE 解析全免：

```typescript
import OpenAI from "openai";

const BASE = "http://127.0.0.1:8642/v1";
const KEY = process.env.API_SERVER_KEY ?? "demo-key-123";

const client = new OpenAI({ baseURL: BASE, apiKey: KEY });

const stream = await client.chat.completions.create({
  model: "hermes-agent",                       // 占位，真正模型由 hermes 配置决定
  messages: [{ role: "user", content: "用一句话介绍你自己，并说出现在几点" }],
  stream: true,
});

process.stdout.write("assistant: ");
for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? "");
}
console.log();
```

运行：`npx tsx demo_a.ts`
预期：逐字流式输出 agent 的自我介绍。

### 7.2 Demo B：`/v1/runs` 结构化事件（零依赖，看工具调用过程）

`demo_b.ts` —— 这条线能看到 agent 的**工具调用过程**（`tool.started` / `tool.completed`），是产品演示「agent 在做什么」的最佳视角。**零依赖**（Node 20+ 原生 fetch）：

```typescript
const BASE = "http://127.0.0.1:8642/v1";
const KEY = process.env.API_SERVER_KEY ?? "demo-key-123";

const brief = (obj: unknown, n = 160): string => {
  if (obj == null || obj === "") return "";
  const s = typeof obj === "string" ? obj : JSON.stringify(obj);
  return s.length <= n ? s : s.slice(0, n) + "…";
};

// 1) 启动 run，立即拿到 run_id（202）
const start = await fetch(`${BASE}/runs`, {
  method: "POST",
  headers: { Authorization: `Bearer ${KEY}`, "Content-Type": "application/json" },
  body: JSON.stringify({
    model: "hermes-agent",
    input: "在当前目录创建 hello.txt，内容写 Hello Hermes，然后读出来给我看",
  }),
});
if (!start.ok) throw new Error(`start failed: ${start.status}`);
const { run_id } = (await start.json()) as { run_id: string };
console.log(`[start] run_id = ${run_id}\n`);

// 2) 订阅该 run 的 SSE 事件流（手写解析，支持 Authorization header）
const resp = await fetch(`${BASE}/runs/${run_id}/events`, {
  headers: { Authorization: `Bearer ${KEY}` },
});
if (!resp.ok || !resp.body) throw new Error(`events ${resp.status}`);

const reader = resp.body.getReader();
const decoder = new TextDecoder();
let buffer = "";
let finished = false;

while (!finished) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });
  let sep: number;
  while ((sep = buffer.indexOf("\n\n")) !== -1) {        // SSE 事件以空行分隔
    const block = buffer.slice(0, sep);
    buffer = buffer.slice(sep + 2);
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("data: ")) data += line.slice(6);
      // 忽略 ': stream closed' 等 SSE comment
    }
    if (!data) continue;
    let p: Record<string, unknown> = {};
    try { p = JSON.parse(data); } catch { /* keep empty */ }
    const event = String(p.event ?? ""); // 关键：事件类型在 JSON 里，不是 event: 前缀
    if (!event) continue;
    switch (event) {
      case "run.started": console.log("▶ run 开始"); break;
      case "message.delta": process.stdout.write(String(p.delta ?? "")); break;
      case "tool.started":
        console.log(`\n  🔧 调用工具: ${p.tool}  ${brief(p.preview)}`); break;
      case "tool.completed": {
        const err = p.error ? " ⚠出错" : "";
        console.log(`\n  ✅ 完成: ${p.tool}  耗时 ${p.duration}s${err}`); break; }
      case "run.completed":
        console.log(`\n■ run 完成。输出: ${brief(p.output)}`); finished = true; break;
      case "run.failed":
      case "run.cancelled":
      case "error":
        console.log(`\n■ ${event}: ${brief(p)}`);
        finished = true; break;
    }
    if (finished) break;
  }
}
```

预期输出：`run.started` → `tool.started(write_file)` → `tool.completed` → `tool.started(read_file)` → … → `run.completed`，agent 真的创建了文件并读回。

### 7.3 运行方式

```bash
# 方式 1：用 tsx（Demo A 带 openai 时方便）
npx tsx demo_b.ts

# 方式 2：Node 22 原生跑 TS（零依赖、零编译，需把扩展名改成 .mts 以启用 ESM + top-level await）
node --experimental-strip-types demo_b.mts
```

### 7.4 为什么 Demo B 要手写 fetch（重要）

`/v1/runs/events` 需要 `Authorization: Bearer` 鉴权，而**浏览器/Node 原生 `EventSource` 不支持自定义 header**（Web 标准限制）。所以订阅事件流必须手写 `fetch` + `ReadableStream` 解析 SSE。

`/v1/chat/completions` 走官方 `openai` SDK 则无此问题——SDK 内部已处理。

---

## 8. 演示话术：用同一套 demo 证明「不支持 AskUserQuestion」

这是本 demo 的核心目的。**对比演示**：

### 步骤 1：构造一个歧义任务

让 agent 处理一个**信息不足、理应反问用户**的任务：

> "帮我把项目部署到服务器。"

在「理想情况」（如 CLI 接入）下，agent 会调用 `clarify` 工具弹出「部署到哪个环境？staging / prod」「服务器地址？」等卡片让用户选。

### 步骤 2：用 Demo B 跑这个任务

把 `demo_b.ts` 里的 `input` 换成歧义任务：

```typescript
input: "帮我把项目部署到服务器。"
```

### 步骤 3：观察事件流——**没有任何 clarify 相关事件**

你会观察到三种现象之一（均证明 AskUserQuestion 缺失）：

| 现象 | 说明 |
|------|------|
| agent 直接假设并尝试（如自己编一个部署命令去执行） | 看不到 clarify 工具，只能「自行假设」——这正是产品要解决的痛点 |
| agent 在**最终文本回复**里反问用户（"请问部署到哪个环境？"） | 这是「文本反问」，不是「结构化卡片」；用户必须开**新一轮**对话才能回答 |
| agent 调用 terminal 工具时触发 `approval.request`（若有危险命令） | **审批卡片能弹，提问卡片不能弹**——这正是文档 01 第 5 节讲的不对称 |

### 给产品的一句话总结

> 「同样的接入方式下，危险命令的审批卡片能弹出来（有 SSE 事件 + 回填端点），但 AskUserQuestion 提问卡片弹不出来——因为底层这条链路压根没给提问能力接线。这个 demo 就是用生产同款接入复现了这个边界。」

---

## 9. 常见问题

| 问题 | 排查 |
|------|------|
| `connection refused` | `hermes gateway run` 是否在前台运行？`API_SERVER_ENABLED=true` 是否 export？端口对不对？ |
| `401 Unauthorized` | 客户端的 `Bearer` token 与 `API_SERVER_KEY` 不一致（demo 代码读 `API_SERVER_KEY` 环境变量，默认 `demo-key-123`）。 |
| 模型不回复 / 报 provider 错误 | Provider 未配置或 key 无效。先跑 `hermes`（不带参数）确认交互配置成功，或检查 `.env` 的 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。 |
| top-level await 报错（`cjs` format） | **必做**：`npm pkg set type=module`（package.json 加 `type:module`）。tsx 与 node 原生都需要 ESM 才支持 top-level await，仅 `npx tsx` 不够（实战验证）。 |
| `fetch is not defined` | Node 版本过低。需 Node 18+（仓库要求 20+，本机 22.15 ✅）。 |
| Demo A 报找不到 `openai` | `cd demo && npm i openai`。 |
| 浏览器页面调不通 | 检查 `API_SERVER_CORS_ORIGINS`；演示用 `*`，生产收紧。 |
| 端口 8642 被占 | `export API_SERVER_PORT=8643` 改端口，客户端 `BASE` 同步改。 |

---

## 10. 清理

```bash
# 停止 hermes（Ctrl+C，gateway run 是前台进程）
# 清理 demo 客户端依赖
rm -rf docs/ask-user-question-analysis/demo/node_modules \
       docs/ask-user-question-analysis/demo/package.json \
       docs/ask-user-question-analysis/demo/package-lock.json
# 清理 hermes 虚拟环境（如不再需要）
rm -rf .venv-demo
# 清理测试生成的文件
rm -f hello.txt
```

---

## 附：核心端点速查（源码：`gateway/platforms/api_server.py:5-21`）

| 方法 | 路径 | 用途 |
|------|------|------|
| POST | `/v1/chat/completions` | OpenAI 兼容 Chat Completions（`stream:true` 走 SSE） |
| POST | `/v1/responses` | OpenAI Responses API（`previous_response_id` 续接） |
| GET | `/v1/responses/{id}` | 取已存储响应 |
| POST | `/v1/runs` | 启动 run，立即返回 `run_id`（202） |
| GET | `/v1/runs/{id}/events` | 该 run 的结构化 SSE 事件流 |
| POST | `/v1/runs/{id}/approval` | **回填审批**（注意：没有 `/clarify` 对应物） |
| POST | `/v1/runs/{id}/stop` | 中断运行中的 agent |
| GET | `/v1/models` | 列出 `hermes-agent` |
| GET | `/v1/capabilities` | 机器可读的能力清单 |
