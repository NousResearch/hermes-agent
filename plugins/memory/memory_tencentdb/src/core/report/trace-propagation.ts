/**
 * Trace Context 跨异步边界传播工具（门面层）
 *
 * 用于在异步任务中序列化/反序列化 OTel Trace Context，
 * 实现 HTTP 请求 → Pipeline Worker 的跨异步链路关联。
 *
 * 使用方式：
 *   // 入队时：序列化当前 Trace Context 到 TaskPayload.data
 *   const traceCtx = serializeTraceContext();
 *   task.data = { ...task.data, ...traceCtx };
 *
 *   // 消费时：从 TaskPayload.data 反序列化恢复 Trace Context
 *   const parentCtx = deserializeTraceContext(task.data);
 *   // 在 parentCtx 中创建 CONSUMER Span
 *
 * 公开 API 签名保持不变，调用方无需修改。
 */

import { getObservabilityBackend } from "./factory.js";

/**
 * 序列化当前 Trace Context 到一个 plain object。
 * 返回的 object 可以直接 spread 到 TaskPayload.data 中。
 *
 * 如果当前没有有效的 Span Context，返回空对象。
 */
export function serializeTraceContext(): Record<string, string | number> {
  try {
    return getObservabilityBackend().tracePropagation.serializeTraceContext();
  } catch {
    return {};
  }
}

/**
 * 从 TaskPayload.data 反序列化恢复 Trace Context。
 *
 * 返回一个 OTel Context，可以用于创建 follow-from link 的 CONSUMER Span。
 * 如果 data 中没有 trace 信息，返回 ROOT_CONTEXT。
 */
export function deserializeTraceContext(
  data?: Record<string, unknown>,
// eslint-disable-next-line @typescript-eslint/no-explicit-any
): { parentContext: any; parentSpanContext: any | null } {
  try {
    return getObservabilityBackend().tracePropagation.deserializeTraceContext(data);
  } catch {
    return { parentContext: {}, parentSpanContext: null };
  }
}
