/**
 * Demo B — /v1/runs 结构化 SSE 事件流（看工具调用过程）。
 *
 * 零依赖（Node 20+ 原生 fetch + ReadableStream）。
 * 运行：
 *   npx tsx demo_b.ts                         # 需 npm i -D tsx
 *   node --experimental-strip-types demo_b.mts # 零依赖原生跑（把扩展名改成 .mts）
 *
 * 为什么手写 fetch：/v1/runs/events 需要 Authorization header，
 *   而原生 EventSource 不支持自定义 header，所以手写 fetch + 流式解析 SSE。
 */
const BASE = "http://127.0.0.1:8642/v1";
const KEY = process.env.API_SERVER_KEY ?? "demo-key-123";

const brief = (obj: unknown, n = 160): string => {
  const s = typeof obj === "string" ? obj : JSON.stringify(obj);
  return s.length <= n ? s : s.slice(0, n) + "…";
};

// 1) 启动 run，立即拿到 run_id（202 返回）
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
  while ((sep = buffer.indexOf("\n\n")) !== -1) {
    // SSE 事件以空行分隔
    const block = buffer.slice(0, sep);
    buffer = buffer.slice(sep + 2);
    let event = "";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) event = line.slice(7).trim();
      else if (line.startsWith("data: ")) data += line.slice(6);
    }
    if (!event || !data) continue;
    let p: Record<string, unknown> = {};
    try {
      p = JSON.parse(data);
    } catch {
      /* keep empty */
    }
    switch (event) {
      case "run.started":
        console.log("▶ run 开始");
        break;
      case "tool.started":
        console.log(`  🔧 调用工具: ${p.tool}  参数: ${brief(p.args)}`);
        break;
      case "tool.completed":
        console.log(`  ✅ 工具完成: ${p.tool}  结果: ${brief(p.result)}`);
        break;
      case "run.completed":
      case "run.failed":
      case "run.cancelled":
      case "error":
        console.log(`■ ${event}: ${brief(p)}`);
        finished = true;
        break;
    }
    if (finished) break;
  }
}
