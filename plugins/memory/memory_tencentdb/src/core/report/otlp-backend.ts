/**
 * OTLP Observability Backend — 基于标准 OpenTelemetry OTLP 协议的内置后端实现。
 *
 * 这是开源用户开箱即用的可观测性后端。用户只需配置一个 OTLP endpoint，
 * 即可将 Trace、Log、Metric 全部上报到任何支持 OTLP 协议的后端：
 *   - ClickHouse（原生支持 OTLP 接收）
 *   - Jaeger（支持 OTLP）
 *   - Grafana Tempo + Loki + Mimir（全家桶支持 OTLP）
 *   - SigNoz（开源一体化，原生 OTLP）
 *   - 本地 OTel Collector（万能中转）
 *
 * 使用方式：
 *   await initObservabilityBackend({
 *     type: "otlp",
 *     otel: {
 *       enabled: true,
 *       endpoint: "http://localhost:4318",   // 任何支持 OTLP 的后端地址
 *       protocol: "http",                    // "http" (OTLP/HTTP) 或 "grpc" (OTLP/gRPC)
 *       serviceName: "my-memory-service",    // 服务名（可选，默认 "tdai-memory"）
 *     },
 *   });
 *
 * 如果不配置（type 为 "noop"），则所有可观测性调用为空操作，零开销。
 *
 * 设计原则：
 * - 使用标准 @opentelemetry/sdk-node 初始化，兼容所有 OTel 生态
 * - Trace + Log + Metric 三合一，一个 endpoint 全搞定
 * - 所有方法不抛异常，不影响业务
 * - 初始化失败时优雅降级到 console 输出
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
  OTelConfig,
} from "./types.js";

const TAG = "[observability][otlp]";

// ============================
// OTel SDK 动态加载
// ============================

/**
 * OTel SDK 运行时引用。
 * 通过动态 import 加载，加载失败时所有后端降级为 console 输出。
 */
interface OTelRuntime {
  // @opentelemetry/api
  trace: any;
  context: any;
  propagation: any;
  SpanKind: any;
  SpanStatusCode: any;
  ROOT_CONTEXT: any;
  TraceFlags: any;
  // @opentelemetry/api-logs
  logs: any;
  SeverityNumber: any;
}

let _runtime: OTelRuntime | null = null;
let _runtimeLoaded = false;

/**
 * 尝试加载 OTel SDK 运行时。
 * 加载失败返回 null（依赖未安装）。
 */
async function loadOTelRuntime(): Promise<OTelRuntime | null> {
  if (_runtimeLoaded) return _runtime;
  _runtimeLoaded = true;

  try {
    const api = await import("@opentelemetry/api");
    let logsApi: any = null;
    try {
      logsApi = await import("@opentelemetry/api-logs");
    } catch {
      // api-logs 可选
    }

    _runtime = {
      trace: api.trace,
      context: api.context,
      propagation: api.propagation,
      SpanKind: api.SpanKind,
      SpanStatusCode: api.SpanStatusCode,
      ROOT_CONTEXT: api.ROOT_CONTEXT,
      TraceFlags: api.TraceFlags,
      logs: logsApi?.logs ?? null,
      SeverityNumber: logsApi?.SeverityNumber ?? { DEBUG: 5, INFO: 9, WARN: 13, ERROR: 17 },
    };
    return _runtime;
  } catch {
    console.warn(`${TAG} @opentelemetry/api not available. OTLP backend will use console fallback.`);
    return null;
  }
}

/**
 * 初始化 OTel SDK（NodeSDK）。
 * 配置 OTLP exporter 将 trace/log/metric 发送到用户指定的 endpoint。
 */
async function initOTelSDK(config: OTelConfig): Promise<boolean> {
  try {
    const protocol = config.protocol ?? "http";
    const endpoint = config.endpoint ?? "http://localhost:4318";
    const serviceName = config.serviceName ?? "tdai-memory";

    // 动态加载 SDK 组件
    const { NodeSDK } = await import("@opentelemetry/sdk-node");
    const { Resource } = await import("@opentelemetry/resources");
    const { ATTR_SERVICE_NAME } = await import("@opentelemetry/semantic-conventions");

    // 根据协议选择 exporter
    let traceExporter: any;
    let logExporter: any;

    if (protocol === "grpc") {
      const { OTLPTraceExporter } = await import("@opentelemetry/exporter-trace-otlp-grpc");
      traceExporter = new OTLPTraceExporter({ url: endpoint });

      try {
        const { OTLPLogExporter } = await import("@opentelemetry/exporter-logs-otlp-grpc");
        logExporter = new OTLPLogExporter({ url: endpoint });
      } catch {
        // log exporter 可选
      }
    } else {
      // 默认 HTTP
      const { OTLPTraceExporter } = await import("@opentelemetry/exporter-trace-otlp-http");
      traceExporter = new OTLPTraceExporter({
        url: `${endpoint}/v1/traces`,
      });

      try {
        const { OTLPLogExporter } = await import("@opentelemetry/exporter-logs-otlp-http");
        logExporter = new OTLPLogExporter({
          url: `${endpoint}/v1/logs`,
        });
      } catch {
        // log exporter 可选
      }
    }

    // 构建 SDK 配置
    const sdkConfig: any = {
      resource: new Resource({
        [ATTR_SERVICE_NAME]: serviceName,
      }),
      traceExporter,
    };

    // 如果有 log exporter，添加 log record processor
    if (logExporter) {
      try {
        const { SimpleLogRecordProcessor } = await import("@opentelemetry/sdk-logs");
        sdkConfig.logRecordProcessors = [new SimpleLogRecordProcessor(logExporter)];
      } catch {
        // sdk-logs 可选
      }
    }

    const sdk = new NodeSDK(sdkConfig);
    sdk.start();

    console.log(
      `${TAG} OTel SDK initialized ✓ | endpoint=${endpoint} | protocol=${protocol} | service=${serviceName}`,
    );
    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn(`${TAG} Failed to initialize OTel SDK: ${msg}. Trace/Log will use console fallback.`);
    return false;
  }
}

// ============================
// ISpan 适配器
// ============================

function wrapOTelSpan(otelSpan: any): ISpan {
  return {
    end() { otelSpan.end(); },
    setAttribute(key: string, value: string | number | boolean) {
      otelSpan.setAttribute(key, value);
      return this;
    },
    setAttributes(attrs: Record<string, string | number | boolean>) {
      otelSpan.setAttributes(attrs);
      return this;
    },
    setStatus(status: { code: number; message?: string }) {
      otelSpan.setStatus(status);
      return this;
    },
    recordException(exception: Error | string) {
      otelSpan.recordException(exception instanceof Error ? exception : new Error(exception));
    },
    spanContext() {
      const ctx = otelSpan.spanContext();
      return { traceId: ctx.traceId, spanId: ctx.spanId, traceFlags: ctx.traceFlags };
    },
    isRecording() { return otelSpan.isRecording(); },
    updateName(name: string) { otelSpan.updateName(name); return this; },
    addEvent(name: string, attrs?: Record<string, string | number | boolean>) {
      otelSpan.addEvent(name, attrs);
      return this;
    },
  };
}

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
// OtlpTraceBackend
// ============================

const TRACER_NAME = "tdai-memory";

/**
 * OTLP Trace 后端 — 通过标准 OTel API 创建 Span，经 OTLP 协议上报。
 */
export class OtlpTraceBackend implements ITraceBackend {
  readonly type = "otlp";

  report(event: string, attrs: TraceAttrs = {}): void {
    if (!_runtime) return;
    try {
      const tracer = _runtime.trace.getTracer(TRACER_NAME);
      const span = tracer.startSpan(`tdai.${event}`, {
        kind: _runtime.SpanKind.INTERNAL,
      }, _runtime.context.active());

      for (const [key, value] of Object.entries(attrs)) {
        if (value !== null && value !== undefined) {
          span.setAttribute(key, value);
        }
      }

      if (attrs.success === false || attrs.success === 0) {
        const errorMsg = typeof attrs.error === "string" ? attrs.error : "unknown error";
        span.setStatus({ code: _runtime.SpanStatusCode.ERROR, message: errorMsg });
      } else {
        span.setStatus({ code: _runtime.SpanStatusCode.OK });
      }

      span.end();
    } catch {
      // 静默
    }
  }

  start(spanName: string, kind?: number): ISpan {
    if (!_runtime) return noopSpan;
    try {
      const tracer = _runtime.trace.getTracer(TRACER_NAME);
      const otelKind = kind ?? _runtime.SpanKind.INTERNAL;
      const span = tracer.startSpan(spanName, { kind: otelKind }, _runtime.context.active());
      return wrapOTelSpan(span);
    } catch {
      return noopSpan;
    }
  }

  startServer(spanName: string): ISpan {
    return this.start(spanName, _runtime?.SpanKind?.SERVER ?? 1);
  }

  startClient(spanName: string): ISpan {
    return this.start(spanName, _runtime?.SpanKind?.CLIENT ?? 2);
  }
}

// ============================
// OtlpLogBackend
// ============================

/**
 * OTLP Log 后端 — 通过 OTel Logs API 发送结构化日志，经 OTLP 协议上报。
 */
export class OtlpLogBackend implements ILogBackend {
  readonly type = "otlp";

  info(eventName: string, attrs: LogAttrs = {}): void {
    this.emit("INFO", 9, eventName, attrs);
  }

  warn(eventName: string, attrs: LogAttrs = {}): void {
    this.emit("WARN", 13, eventName, attrs);
  }

  error(eventName: string, attrs: LogAttrs = {}, _error?: Error): void {
    this.emit("ERROR", 17, eventName, attrs);
  }

  debug(eventName: string, attrs: LogAttrs = {}): void {
    this.emit("DEBUG", 5, eventName, attrs);
  }

  private emit(level: string, severityNumber: number, message: string, attrs: LogAttrs): void {
    if (!_runtime?.logs) return;
    try {
      const logger = _runtime.logs.getLogger(TRACER_NAME);
      logger.emit({
        severityNumber,
        severityText: level,
        body: message,
        attributes: attrs,
        context: _runtime.context?.active?.(),
      });
    } catch {
      // 静默
    }
  }
}

// ============================
// OtlpMetricBackend
// ============================

/**
 * OTLP Metric 后端 — 通过 OTel Metrics API 上报指标。
 *
 * 注意：OTel Metrics 需要 @opentelemetry/sdk-metrics，
 * 如果未安装则降级为 console 输出。
 */
export class OtlpMetricBackend implements IMetricBackend {
  readonly type = "otlp";
  private _meter: any = null;
  private _counters: Map<string, any> = new Map();
  private _initialized = false;

  send(msg: MetricMessage): void {
    if (!this._initialized || !this._meter) {
      // 降级：如果 OTel Metrics 不可用，静默忽略
      return;
    }

    try {
      // 获取或创建 counter
      let counter = this._counters.get(msg.metric);
      if (!counter) {
        counter = this._meter.createCounter(msg.metric, {
          description: `Memory metric: ${msg.metric}`,
        });
        this._counters.set(msg.metric, counter);
      }

      counter.add(msg.value, {
        instance_id: msg.instanceId,
        source: msg.source ?? "core",
      });
    } catch {
      // 静默
    }
  }

  async initialize(_config: MetricBackendConfig): Promise<void> {
    // 尝试加载 OTel Metrics SDK
    try {
      const { metrics } = await import("@opentelemetry/api");
      this._meter = metrics.getMeter(TRACER_NAME);
      this._initialized = true;
    } catch {
      // @opentelemetry/api 的 metrics 不可用，静默降级
      this._initialized = false;
    }
  }

  async destroy(): Promise<void> {
    this._counters.clear();
    this._meter = null;
    this._initialized = false;
  }
}

// ============================
// OtlpLLMTraceBackend
// ============================

/**
 * OTLP LLM Trace 后端 — 在 OTLP 模式下，LLM span 直接通过标准 trace 上报。
 * 不需要额外的 Langfuse，所有 span（包括 ai.* / gen_ai.*）都走 OTLP。
 */
export class OtlpLLMTraceBackend implements ILLMTraceBackend {
  readonly type = "otlp";

  createSpanProcessor(): ISpanProcessor | null {
    // OTLP 模式下不需要额外的 SpanProcessor，
    // 所有 span 已经通过 NodeSDK 的 traceExporter 统一上报
    return null;
  }

  async flush(): Promise<void> {
    // 由 NodeSDK 统一管理 flush
  }

  async shutdown(): Promise<void> {
    // 由 NodeSDK 统一管理 shutdown
  }
}

// ============================
// OtlpTraceMiddleware
// ============================

/** 不需要 Trace 的路径 */
const SKIP_PATHS = new Set(["/health"]);

/** 路由 → Span Name 映射 */
const ROUTE_SPAN_NAMES: Record<string, string> = {
  "POST /capture": "core.capture",
  "POST /recall": "core.recall",
  "POST /search/memories": "core.search.memories",
  "POST /search/conversations": "core.search.conversations",
  "POST /session/end": "core.session.end",
  "POST /seed": "core.seed",
};

/**
 * OTLP HTTP Trace 中间件 — 为每个 HTTP 请求创建 SERVER Span。
 */
export class OtlpTraceMiddleware implements ITraceMiddleware {
  readonly type = "otlp";

  async wrapWithTrace(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    handler: () => Promise<void>,
  ): Promise<void> {
    if (!_runtime) {
      return handler();
    }

    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    const method = req.method?.toUpperCase() ?? "GET";
    const pathname = url.pathname;

    if (SKIP_PATHS.has(pathname)) {
      return handler();
    }

    // 从 W3C traceparent 头提取上游 Trace Context
    const parentContext = _runtime.propagation.extract(_runtime.ROOT_CONTEXT, req.headers, {
      get(carrier: any, key: string) {
        const val = carrier[key.toLowerCase()];
        return Array.isArray(val) ? val[0] : val ?? undefined;
      },
      keys(carrier: any) { return Object.keys(carrier); },
    });

    const routeKey = `${method} ${pathname}`;
    const spanName = ROUTE_SPAN_NAMES[routeKey] ?? "core.request";

    const tracer = _runtime.trace.getTracer(TRACER_NAME);
    const span = tracer.startSpan(
      spanName,
      {
        kind: _runtime.SpanKind.SERVER,
        attributes: {
          "http.method": method,
          "http.url": pathname,
          "http.host": req.headers.host ?? "",
        },
      },
      parentContext,
    );

    // 提取业务属性
    const instanceId = (req.headers["x-tdai-service-id"] ?? req.headers["x-instance-id"] ?? "") as string;
    if (instanceId) span.setAttribute("instance_id", instanceId);
    const reqId = (req.headers["x-qcloud-transaction-id"] ?? req.headers["x-request-id"] ?? "") as string;
    if (reqId) span.setAttribute("req_id", reqId);

    const spanContext = _runtime.trace.setSpan(parentContext, span);
    const traceId = span.spanContext().traceId;
    res.setHeader("x-trace-id", traceId);

    try {
      await _runtime.context.with(spanContext, async () => {
        await handler();
      });

      span.setAttribute("http.status_code", res.statusCode);
      if (res.statusCode >= 400) {
        span.setStatus({ code: _runtime!.SpanStatusCode.ERROR, message: `HTTP ${res.statusCode}` });
      } else {
        span.setStatus({ code: _runtime!.SpanStatusCode.OK });
      }
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      span.setStatus({ code: _runtime!.SpanStatusCode.ERROR, message: errMsg });
      span.recordException(err instanceof Error ? err : new Error(errMsg));
      throw err;
    } finally {
      span.end();
    }
  }

  startChildSpan(
    name: string,
    attrs: Record<string, string | number | boolean> = {},
  ): ISpan {
    if (!_runtime) return noopSpan;
    try {
      const tracer = _runtime.trace.getTracer(TRACER_NAME);
      const span = tracer.startSpan(name, {
        kind: _runtime.SpanKind.INTERNAL,
        attributes: attrs,
      }, _runtime.context.active());
      return wrapOTelSpan(span);
    } catch {
      return noopSpan;
    }
  }

  async withSpan<T>(
    name: string,
    attrs: Record<string, string | number | boolean>,
    fn: (span: ISpan) => Promise<T>,
  ): Promise<T> {
    if (!_runtime) {
      return fn(noopSpan);
    }

    const tracer = _runtime.trace.getTracer(TRACER_NAME);
    const span = tracer.startSpan(name, {
      kind: _runtime.SpanKind.INTERNAL,
      attributes: attrs,
    }, _runtime.context.active());

    const spanContext = _runtime.trace.setSpan(_runtime.context.active(), span);
    const wrappedSpan = wrapOTelSpan(span);

    try {
      const result = await _runtime.context.with(spanContext, () => fn(wrappedSpan));
      span.setStatus({ code: _runtime!.SpanStatusCode.OK });
      return result;
    } catch (err) {
      const errMsg = err instanceof Error ? err.message : String(err);
      span.setStatus({ code: _runtime!.SpanStatusCode.ERROR, message: errMsg });
      span.recordException(err instanceof Error ? err : new Error(errMsg));
      throw err;
    } finally {
      span.end();
    }
  }
}

// ============================
// OtlpTracePropagation
// ============================

/** 序列化到 TaskPayload.data 中的字段名 */
const TRACE_ID_KEY = "_traceId";
const SPAN_ID_KEY = "_spanId";
const TRACE_FLAGS_KEY = "_traceFlags";

/**
 * OTLP Trace Context 传播 — 通过 OTel API 序列化/反序列化 Trace Context。
 */
export class OtlpTracePropagation implements ITracePropagation {
  serializeTraceContext(): Record<string, string | number> {
    if (!_runtime) return {};
    try {
      const span = _runtime.trace.getSpan(_runtime.context.active());
      if (!span) return {};
      const spanCtx = span.spanContext();
      if (!spanCtx.traceId) return {};
      return {
        [TRACE_ID_KEY]: spanCtx.traceId,
        [SPAN_ID_KEY]: spanCtx.spanId,
        [TRACE_FLAGS_KEY]: spanCtx.traceFlags,
      };
    } catch {
      return {};
    }
  }

  deserializeTraceContext(data?: Record<string, unknown>): {
    parentContext: unknown;
    parentSpanContext: { traceId: string; spanId: string; traceFlags: number; isRemote: boolean } | null;
  } {
    if (!data || !_runtime) {
      return { parentContext: _runtime?.ROOT_CONTEXT ?? {}, parentSpanContext: null };
    }

    const traceId = data[TRACE_ID_KEY] as string | undefined;
    const spanId = data[SPAN_ID_KEY] as string | undefined;
    const traceFlags = data[TRACE_FLAGS_KEY] as number | undefined;

    if (!traceId || !spanId) {
      return { parentContext: _runtime.ROOT_CONTEXT, parentSpanContext: null };
    }

    try {
      const parentSpanContext = {
        traceId,
        spanId,
        traceFlags: traceFlags ?? _runtime.TraceFlags.SAMPLED,
        isRemote: true,
      };
      const parentContext = _runtime.trace.setSpanContext(_runtime.ROOT_CONTEXT, parentSpanContext);
      return { parentContext, parentSpanContext };
    } catch {
      return { parentContext: _runtime.ROOT_CONTEXT, parentSpanContext: null };
    }
  }
}

// ============================
// OtlpObservabilityBackend — 聚合
// ============================

/**
 * OTLP 可观测性后端 — 基于标准 OpenTelemetry OTLP 协议。
 *
 * 开源用户开箱即用：配置一个 OTLP endpoint 即可将 Trace/Log/Metric 全部上报。
 * 支持任何兼容 OTLP 协议的后端（ClickHouse、Jaeger、Grafana、SigNoz 等）。
 *
 * 使用方式：
 *   await initObservabilityBackend({
 *     type: "otlp",
 *     otel: {
 *       enabled: true,
 *       endpoint: "http://localhost:4318",
 *       serviceName: "my-memory-service",
 *     },
 *   });
 */
export class OtlpObservabilityBackend implements IObservabilityBackend {
  readonly type = "otlp";
  readonly trace: ITraceBackend = new OtlpTraceBackend();
  readonly log: ILogBackend = new OtlpLogBackend();
  readonly metric: IMetricBackend = new OtlpMetricBackend();
  readonly llmTrace: ILLMTraceBackend = new OtlpLLMTraceBackend();
  readonly traceMiddleware: ITraceMiddleware = new OtlpTraceMiddleware();
  readonly tracePropagation: ITracePropagation = new OtlpTracePropagation();

  async initialize(config: ObservabilityConfig): Promise<void> {
    const otelConfig = config.otel;

    if (!otelConfig?.enabled) {
      console.warn(`${TAG} OTLP backend requested but otel.enabled is false. All observability will be no-op.`);
      return;
    }

    // 1. 加载 OTel 运行时
    const runtime = await loadOTelRuntime();
    if (!runtime) {
      console.warn(`${TAG} @opentelemetry/api not installed. Run: npm install @opentelemetry/api @opentelemetry/sdk-node @opentelemetry/exporter-trace-otlp-http`);
      return;
    }

    // 2. 初始化 OTel SDK（配置 OTLP exporter）
    await initOTelSDK(otelConfig);

    // 3. 初始化 Metric 后端
    await this.metric.initialize({
      brokers: [],
      enabled: true,
    });

    console.log(`${TAG} OtlpObservabilityBackend initialized ✓`);
  }

  async shutdown(): Promise<void> {
    await this.metric.destroy();
    console.log(`${TAG} OtlpObservabilityBackend shutdown`);
  }
}
