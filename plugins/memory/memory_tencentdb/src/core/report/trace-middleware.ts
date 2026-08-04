/**
 * Core HTTP Trace 中间件 — 非侵入式 Trace 埋点（门面层）
 *
 * 为每个 HTTP 请求创建 SERVER 类型的入口 Span（core.request），
 * 从 traceparent 头恢复上游 Trace Context，实现跨服务链路关联。
 *
 * 使用方式（在 server.ts 中）：
 *   import { wrapWithTrace } from "../core/report/trace-middleware.js";
 *   // 在 createServer 时包装 handleRequest
 *   this.server = http.createServer((req, res) => wrapWithTrace(req, res, () => this.handleRequest(req, res)));
 *
 * 不修改任何业务代码，纯可观测性组件。
 * 公开 API 签名保持不变，调用方无需修改。
 */

import http from "node:http";
import { getObservabilityBackend } from "./factory.js";
import type { ISpan } from "./types.js";

// 重导出 Span 类型（保持向后兼容）
export type { Span } from "@opentelemetry/api";

/**
 * 包装 HTTP 请求处理器，添加 Trace 埋点。
 *
 * @param req HTTP 请求
 * @param res HTTP 响应
 * @param handler 原始请求处理器
 */
export async function wrapWithTrace(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  handler: () => Promise<void>,
): Promise<void> {
  try {
    return await getObservabilityBackend().traceMiddleware.wrapWithTrace(req, res, handler);
  } catch (err) {
    // 如果是 handler 本身抛出的异常，需要重新抛出
    throw err;
  }
}

/** Noop Span — 后端不可用时的安全替代 */
const noopSpan: ISpan = {
  end() {},
  setAttribute() { return this; },
  setAttributes() { return this; },
  setStatus() { return this; },
  recordException() {},
  spanContext() { return { traceId: "", spanId: "", traceFlags: 0 }; },
  isRecording() { return false; },
  updateName() { return this; },
  addEvent() { return this; },
};

/**
 * 创建一个子 Span（用于在业务处理器内部创建更细粒度的 Span）。
 *
 * @param name Span 名称（如 "core.vdb.write"）
 * @param attrs Span 属性
 * @returns Span 实例，调用方需要手动 span.end()
 */
export function startChildSpan(
  name: string,
  attrs: Record<string, string | number | boolean> = {},
// eslint-disable-next-line @typescript-eslint/no-explicit-any
): any {
  try {
    return getObservabilityBackend().traceMiddleware.startChildSpan(name, attrs);
  } catch {
    return noopSpan;
  }
}

/**
 * 在当前 Span context 中执行一个函数，并自动创建子 Span。
 *
 * @param name Span 名称
 * @param attrs Span 属性
 * @param fn 要执行的函数
 * @returns fn 的返回值
 */
export async function withSpan<T>(
  name: string,
  attrs: Record<string, string | number | boolean>,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  fn: (span: any) => Promise<T>,
): Promise<T> {
  try {
    return await getObservabilityBackend().traceMiddleware.withSpan(name, attrs, fn);
  } catch (err) {
    // 如果是 fn 本身抛出的异常，需要重新抛出
    throw err;
  }
}