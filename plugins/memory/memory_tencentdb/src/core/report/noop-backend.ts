/**
 * Noop Observability Backend — 空操作实现。
 *
 * 所有方法为空操作，不产生任何副作用。
 * 用于开源环境下未配置任何可观测性后端时的默认实现。
 *
 * 设计原则：
 * - 所有方法不抛异常
 * - 不产生任何 I/O 或副作用
 * - 零性能开销
 */

import type http from "node:http";
import type {
  ITraceBackend,
  ILogBackend,
  IMetricBackend,
  ILLMTraceBackend,
  ITraceMiddleware,
  ITracePropagation,
  IObservabilityBackend,
  ISpan,
  ISpanProcessor,
  TraceAttrs,
  LogAttrs,
  MetricMessage,
  MetricBackendConfig,
  ObservabilityConfig,
} from "./types.js";

// ============================
// Noop Span
// ============================

/** 空操作 Span — 所有方法为 no-op */
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

// ============================
// Noop SpanProcessor
// ============================

/** 空操作 SpanProcessor */
const noopSpanProcessor: ISpanProcessor = {
  onStart() {},
  onEnd() {},
  async forceFlush() {},
  async shutdown() {},
};

// ============================
// NoopTraceBackend
// ============================

export class NoopTraceBackend implements ITraceBackend {
  readonly type = "noop";

  report(_event: string, _attrs?: TraceAttrs): void {
    // no-op
  }

  start(_spanName: string, _kind?: number): ISpan {
    return noopSpan;
  }

  startServer(_spanName: string): ISpan {
    return noopSpan;
  }

  startClient(_spanName: string): ISpan {
    return noopSpan;
  }
}

// ============================
// NoopLogBackend
// ============================

export class NoopLogBackend implements ILogBackend {
  readonly type = "noop";

  info(_eventName: string, _attrs?: LogAttrs): void {
    // no-op
  }

  warn(_eventName: string, _attrs?: LogAttrs): void {
    // no-op
  }

  error(_eventName: string, _attrs?: LogAttrs, _error?: Error): void {
    // no-op
  }

  debug(_eventName: string, _attrs?: LogAttrs): void {
    // no-op
  }
}

// ============================
// NoopMetricBackend
// ============================

export class NoopMetricBackend implements IMetricBackend {
  readonly type = "noop";

  send(_msg: MetricMessage): void {
    // no-op
  }

  async initialize(_config: MetricBackendConfig): Promise<void> {
    // no-op
  }

  async destroy(): Promise<void> {
    // no-op
  }
}

// ============================
// NoopLLMTraceBackend
// ============================

export class NoopLLMTraceBackend implements ILLMTraceBackend {
  readonly type = "noop";

  createSpanProcessor(): ISpanProcessor | null {
    return noopSpanProcessor;
  }

  async flush(): Promise<void> {
    // no-op
  }

  async shutdown(): Promise<void> {
    // no-op
  }
}

// ============================
// NoopTraceMiddleware
// ============================

export class NoopTraceMiddleware implements ITraceMiddleware {
  readonly type = "noop";

  async wrapWithTrace(
    _req: http.IncomingMessage,
    _res: http.ServerResponse,
    handler: () => Promise<void>,
  ): Promise<void> {
    // 直接透传到原始 handler
    return handler();
  }

  startChildSpan(
    _name: string,
    _attrs?: Record<string, string | number | boolean>,
  ): ISpan {
    return noopSpan;
  }

  async withSpan<T>(
    _name: string,
    _attrs: Record<string, string | number | boolean>,
    fn: (span: ISpan) => Promise<T>,
  ): Promise<T> {
    return fn(noopSpan);
  }
}

// ============================
// NoopTracePropagation
// ============================

export class NoopTracePropagation implements ITracePropagation {
  serializeTraceContext(): Record<string, string | number> {
    return {};
  }

  deserializeTraceContext(_data?: Record<string, unknown>): {
    parentContext: unknown;
    parentSpanContext: { traceId: string; spanId: string; traceFlags: number; isRemote: boolean } | null;
  } {
    return { parentContext: {}, parentSpanContext: null };
  }
}

// ============================
// NoopObservabilityBackend — 聚合
// ============================

/**
 * 空操作可观测性后端 — 所有子后端均为 no-op。
 * 开源环境下的默认实现。
 */
export class NoopObservabilityBackend implements IObservabilityBackend {
  readonly type = "noop";
  readonly trace: ITraceBackend = new NoopTraceBackend();
  readonly log: ILogBackend = new NoopLogBackend();
  readonly metric: IMetricBackend = new NoopMetricBackend();
  readonly llmTrace: ILLMTraceBackend = new NoopLLMTraceBackend();
  readonly traceMiddleware: ITraceMiddleware = new NoopTraceMiddleware();
  readonly tracePropagation: ITracePropagation = new NoopTracePropagation();

  async initialize(_config: ObservabilityConfig): Promise<void> {
    // no-op
  }

  async shutdown(): Promise<void> {
    // no-op
  }
}
