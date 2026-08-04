/**
 * Trace 埋点门面 — 事件即 Span + 传统 Start/End
 *
 * 使用方式：
 *
 *   import { trace } from "./core/report/trace.js";
 *
 *   // 事件即 Span（一行搞定，最常用）
 *   trace.report("l1_extraction", {
 *     sessionKey,
 *     memoriesExtracted: extracted.length,
 *     totalDurationMs: Date.now() - startMs,
 *     success: true,
 *     error: null,
 *   });
 *
 *   // 传统 Start/End（跨服务调用链场景）
 *   const span = trace.start("memory.recall");
 *   // ... 业务逻辑 ...
 *   span.end();
 *
 * 本模块为门面层，内部委托给 ITraceBackend（通过全局单例获取）。
 * 公开 API 签名保持不变，调用方无需修改。
 */

import { getObservabilityBackend } from "./factory.js";
import type { TraceAttrs, ISpan } from "./types.js";

// 从 @opentelemetry/api 重导出 Span 类型（仅类型，不影响运行时）
export type { Span } from "@opentelemetry/api";

export type { TraceAttrs } from "./types.js";

/**
 * 上报一个业务事件（事件即 Span）。
 *
 * 内部创建一个 Span，将 attrs 中的每个字段设为 Span Attribute，
 * 根据 "success" 字段设置 Span Status，然后立即 End。
 */
function report(event: string, attrs: TraceAttrs = {}): void {
  try {
    getObservabilityBackend().trace.report(event, attrs);
  } catch {
    // 静默失败，不阻塞业务
  }
}

/**
 * 创建一个传统的 Span（用于跨服务调用链场景）。
 * 调用方需要手动 span.end()。
 */
function start(spanName: string, kind?: number): ISpan {
  try {
    return getObservabilityBackend().trace.start(spanName, kind);
  } catch {
    return noopSpan;
  }
}

/**
 * 创建一个 SERVER 类型的 Span。
 */
function startServer(spanName: string): ISpan {
  try {
    return getObservabilityBackend().trace.startServer(spanName);
  } catch {
    return noopSpan;
  }
}

/**
 * 创建一个 CLIENT 类型的 Span。
 */
function startClient(spanName: string): ISpan {
  try {
    return getObservabilityBackend().trace.startClient(spanName);
  } catch {
    return noopSpan;
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

export const trace = {
  report,
  start,
  startServer,
  startClient,
};
