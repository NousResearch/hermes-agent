/**
 * OTel context helpers。
 *
 * 背景（为什么需要它）：
 *   skill 抽取 worker 的 `runLoop()` 是在某个 HTTP 请求 handler 里被**懒启动**的
 *   （`resolveConversationAdd` → `wireConversationAdd` → `worker.start()`）。
 *   OTel 用 AsyncLocalStorageContextManager 传播上下文——一个**永不退出**的
 *   `runLoop()` 会永久继承"启动那一刻"的 active span（即那次请求的 span），
 *   于是之后每一次 LLM `generateText` 都成了那条请求 trace 的子 span，被
 *   Langfuse 合并成一条（tags 跨多个 agent、sessionId 混乱）。
 *
 *   `runInRootContext` 把 fn 放到 OTel ROOT_CONTEXT 里执行，切断这种寄生，
 *   使 fn 内新建的 span（如 ai.generateText）成为独立 root（各自 traceId）。
 *
 * 防御式加载 @opentelemetry/api：包缺失（开源/精简部署）时降级为直接执行 fn，
 * 语义等价，不报错。加载方式对齐 otel-sdk-init.ts。
 */

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _context: any = null;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let _rootContext: any = null;

try {
  const api = await import("@opentelemetry/api");
  _context = api.context;
  _rootContext = api.ROOT_CONTEXT;
} catch {
  // @opentelemetry/api 不可用 → 保持 null，runInRootContext 降级为直接执行 fn
}

/**
 * 在 OTel ROOT_CONTEXT 中执行 fn，切断对调用点 active span 的继承。
 *
 * 典型用途：启动永不退出的后台循环时包一层，避免循环把"启动时"的 trace
 * 上下文一直带下去。otel 不可用时直接执行 fn（无 context 隔离，但行为等价）。
 */
export function runInRootContext<T>(fn: () => T): T {
  if (_context && _rootContext) {
    return _context.with(_rootContext, fn) as T;
  }
  return fn();
}
