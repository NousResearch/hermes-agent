/**
 * Console Observability Backend — 控制台输出实现。
 *
 * 将所有可观测性数据输出到 stdout/stderr，用于开发调试。
 * 所有方法都是安全的（不抛异常、不阻塞业务）。
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

const TAG = "[observability][console]";

// ============================
// Console Span
// ============================

/** Console Span — 输出 Span 生命周期到 stdout */
class ConsoleSpan implements ISpan {
  private name: string;
  private attrs: Record<string, string | number | boolean> = {};
  private startTime = Date.now();

  constructor(name: string, attrs?: Record<string, string | number | boolean>) {
    this.name = name;
    if (attrs) this.attrs = { ...attrs };
  }

  end(): void {
    const durationMs = Date.now() - this.startTime;
    console.log(`${TAG}[trace] SPAN_END name=${this.name} duration=${durationMs}ms attrs=${JSON.stringify(this.attrs)}`);
  }

  setAttribute(key: string, value: string | number | boolean): this {
    this.attrs[key] = value;
    return this;
  }

  setAttributes(attrs: Record<string, string | number | boolean>): this {
    Object.assign(this.attrs, attrs);
    return this;
  }

  setStatus(_status: { code: number; message?: string }): this {
    return this;
  }

  recordException(exception: Error | string): void {
    const msg = exception instanceof Error ? exception.message : exception;
    console.error(`${TAG}[trace] EXCEPTION span=${this.name} error=${msg}`);
  }

  spanContext(): { traceId: string; spanId: string; traceFlags: number } {
    return { traceId: "console-trace-id", spanId: "console-span-id", traceFlags: 1 };
  }

  isRecording(): boolean {
    return true;
  }

  updateName(name: string): this {
    this.name = name;
    return this;
  }

  addEvent(name: string, attrs?: Record<string, string | number | boolean>): this {
    console.log(`${TAG}[trace] EVENT span=${this.name} event=${name} attrs=${JSON.stringify(attrs ?? {})}`);
    return this;
  }
}

// ============================
// ConsoleTraceBackend
// ============================

export class ConsoleTraceBackend implements ITraceBackend {
  readonly type = "console";

  report(event: string, attrs: TraceAttrs = {}): void {
    try {
      const filtered: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(attrs)) {
        if (v !== null && v !== undefined) filtered[k] = v;
      }
      console.log(`${TAG}[trace] REPORT event=tdai.${event} attrs=${JSON.stringify(filtered)}`);
    } catch {
      // 静默
    }
  }

  start(spanName: string, _kind?: number): ISpan {
    console.log(`${TAG}[trace] SPAN_START name=${spanName}`);
    return new ConsoleSpan(spanName);
  }

  startServer(spanName: string): ISpan {
    console.log(`${TAG}[trace] SPAN_START name=${spanName} kind=SERVER`);
    return new ConsoleSpan(spanName);
  }

  startClient(spanName: string): ISpan {
    console.log(`${TAG}[trace] SPAN_START name=${spanName} kind=CLIENT`);
    return new ConsoleSpan(spanName);
  }
}

// ============================
// ConsoleLogBackend
// ============================

export class ConsoleLogBackend implements ILogBackend {
  readonly type = "console";

  info(eventName: string, attrs: LogAttrs = {}): void {
    console.info(`${TAG}[log][INFO] ${eventName}`, attrs);
  }

  warn(eventName: string, attrs: LogAttrs = {}): void {
    console.warn(`${TAG}[log][WARN] ${eventName}`, attrs);
  }

  error(eventName: string, attrs: LogAttrs = {}, error?: Error): void {
    if (error) {
      console.error(`${TAG}[log][ERROR] ${eventName}`, { ...attrs, "error.message": error.message, "error.type": error.name });
    } else {
      console.error(`${TAG}[log][ERROR] ${eventName}`, attrs);
    }
  }

  debug(eventName: string, attrs: LogAttrs = {}): void {
    console.debug(`${TAG}[log][DEBUG] ${eventName}`, attrs);
  }
}

// ============================
// ConsoleMetricBackend
// ============================

export class ConsoleMetricBackend implements IMetricBackend {
  readonly type = "console";

  send(msg: MetricMessage): void {
    console.log(`${TAG}[metric] SEND metric=${msg.metric} instance=${msg.instanceId} value=${msg.value}`);
  }

  async initialize(_config: MetricBackendConfig): Promise<void> {
    console.log(`${TAG}[metric] Initialized (console mode)`);
  }

  async destroy(): Promise<void> {
    console.log(`${TAG}[metric] Destroyed (console mode)`);
  }
}

// ============================
// ConsoleLLMTraceBackend
// ============================

export class ConsoleLLMTraceBackend implements ILLMTraceBackend {
  readonly type = "console";

  createSpanProcessor(): ISpanProcessor | null {
    // 返回一个简单的 console processor
    return {
      onStart(_span: unknown) {
        // no-op on start
      },
      onEnd(span: unknown) {
        try {
          const s = span as { name?: string };
          console.log(`${TAG}[llm-trace] SPAN_END name=${s.name ?? "unknown"}`);
        } catch {
          // 静默
        }
      },
      async forceFlush() {},
      async shutdown() {},
    };
  }

  async flush(): Promise<void> {
    console.log(`${TAG}[llm-trace] Flushed (console mode)`);
  }

  async shutdown(): Promise<void> {
    console.log(`${TAG}[llm-trace] Shutdown (console mode)`);
  }
}

// ============================
// ConsoleTraceMiddleware
// ============================

export class ConsoleTraceMiddleware implements ITraceMiddleware {
  readonly type = "console";

  async wrapWithTrace(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    handler: () => Promise<void>,
  ): Promise<void> {
    const method = req.method ?? "GET";
    const url = req.url ?? "/";
    const startTime = Date.now();

    console.log(`${TAG}[middleware] REQUEST_START ${method} ${url}`);

    try {
      await handler();
      const durationMs = Date.now() - startTime;
      console.log(`${TAG}[middleware] REQUEST_END ${method} ${url} status=${res.statusCode} duration=${durationMs}ms`);
    } catch (err) {
      const durationMs = Date.now() - startTime;
      const errMsg = err instanceof Error ? err.message : String(err);
      console.error(`${TAG}[middleware] REQUEST_ERROR ${method} ${url} error=${errMsg} duration=${durationMs}ms`);
      throw err;
    }
  }

  startChildSpan(
    name: string,
    attrs?: Record<string, string | number | boolean>,
  ): ISpan {
    console.log(`${TAG}[middleware] CHILD_SPAN name=${name}`);
    return new ConsoleSpan(name, attrs);
  }

  async withSpan<T>(
    name: string,
    attrs: Record<string, string | number | boolean>,
    fn: (span: ISpan) => Promise<T>,
  ): Promise<T> {
    const span = new ConsoleSpan(name, attrs);
    console.log(`${TAG}[middleware] WITH_SPAN name=${name}`);
    try {
      const result = await fn(span);
      span.end();
      return result;
    } catch (err) {
      span.recordException(err instanceof Error ? err : new Error(String(err)));
      span.end();
      throw err;
    }
  }
}

// ============================
// ConsoleTracePropagation
// ============================

export class ConsoleTracePropagation implements ITracePropagation {
  serializeTraceContext(): Record<string, string | number> {
    return { _traceId: "console-trace-id", _spanId: "console-span-id", _traceFlags: 1 };
  }

  deserializeTraceContext(_data?: Record<string, unknown>): {
    parentContext: unknown;
    parentSpanContext: { traceId: string; spanId: string; traceFlags: number; isRemote: boolean } | null;
  } {
    return { parentContext: {}, parentSpanContext: null };
  }
}

// ============================
// ConsoleObservabilityBackend — 聚合
// ============================

/**
 * Console 可观测性后端 — 所有数据输出到 stdout/stderr。
 * 用于开发调试环境。
 */
export class ConsoleObservabilityBackend implements IObservabilityBackend {
  readonly type = "console";
  readonly trace: ITraceBackend = new ConsoleTraceBackend();
  readonly log: ILogBackend = new ConsoleLogBackend();
  readonly metric: IMetricBackend = new ConsoleMetricBackend();
  readonly llmTrace: ILLMTraceBackend = new ConsoleLLMTraceBackend();
  readonly traceMiddleware: ITraceMiddleware = new ConsoleTraceMiddleware();
  readonly tracePropagation: ITracePropagation = new ConsoleTracePropagation();

  async initialize(_config: ObservabilityConfig): Promise<void> {
    console.log(`${TAG} ConsoleObservabilityBackend initialized`);
  }

  async shutdown(): Promise<void> {
    console.log(`${TAG} ConsoleObservabilityBackend shutdown`);
  }
}
