/**
 * Observability Abstraction Layer — Core Types & Interfaces.
 *
 * This module defines the observability contracts for Trace, Log, Metric,
 * LLM Trace, and HTTP Trace Middleware.
 *
 * Design principles:
 * 1. **Backend-agnostic**: Upper layers (trace.ts, obs-logger.ts, etc.) depend
 *    only on these interfaces — never on OTel SDK, Kafka, or Langfuse directly.
 * 2. **Async-first**: All lifecycle methods return Promises; hot-path methods
 *    (report, send) are synchronous for zero-overhead.
 * 3. **Extensible**: Interface is minimal for v1; implementations can add
 *    backend-specific features without changing the contract.
 * 4. **Safe by default**: All implementations must be error-silent — never
 *    throw exceptions that could affect business logic.
 *
 * Relationship to IStorageBackend (src/core/storage/types.ts):
 *   - IStorageBackend = file storage abstraction (L2/L3 files → COS/local-fs)
 *   - IObservabilityBackend = observability abstraction (Trace/Log/Metric → OTel/Kafka/Langfuse)
 *   Both follow the same pattern: interface + factory + dynamic import for private impl.
 */

import type http from "node:http";

// ============================
// Common Types
// ============================

/** Trace 属性 — 支持基本类型和 null/undefined（会被过滤） */
export type TraceAttrs = Record<string, string | number | boolean | null | undefined>;

/** Log 属性 — 只支持基本类型 */
export type LogAttrs = Record<string, string | number | boolean>;

/** Span 接口 — 与 @opentelemetry/api Span 兼容的最小子集 */
export interface ISpan {
  /** 结束 Span */
  end(): void;
  /** 设置单个属性 */
  setAttribute(key: string, value: string | number | boolean): this;
  /** 批量设置属性 */
  setAttributes(attrs: Record<string, string | number | boolean>): this;
  /** 设置 Span 状态 */
  setStatus(status: { code: number; message?: string }): this;
  /** 记录异常 */
  recordException(exception: Error | string): void;
  /** 获取 Span Context */
  spanContext(): { traceId: string; spanId: string; traceFlags: number };
  /** 是否正在记录 */
  isRecording(): boolean;
  /** 更新 Span 名称 */
  updateName(name: string): this;
  /** 添加事件 */
  addEvent(name: string, attrs?: Record<string, string | number | boolean>): this;
}

/** SpanProcessor 接口 — 与 @opentelemetry/sdk-trace-base SpanProcessor 兼容的最小子集 */
export interface ISpanProcessor {
  onStart(span: unknown, parentContext: unknown): void;
  onEnd(span: unknown): void;
  forceFlush(): Promise<void>;
  shutdown(): Promise<void>;
}

// ============================
// ITraceBackend — Trace 抽象
// ============================

/**
 * Trace 后端接口。
 *
 * 实现：
 * - NoopTraceBackend   — 空操作（开源默认）
 * - ConsoleTraceBackend — stdout 输出（开发调试）
 * - OTelTraceBackend   — OpenTelemetry（内部：智研 + ClickHouse 双写）
 */
export interface ITraceBackend {
  /** 后端标识 */
  readonly type: string;

  /**
   * 上报一个业务事件（事件即 Span）。
   * 内部创建 Span → 设置属性 → 设置状态 → End。
   */
  report(event: string, attrs?: TraceAttrs): void;

  /**
   * 创建一个传统 Span（调用方需手动 span.end()）。
   * @param spanName Span 名称
   * @param kind SpanKind 数值（INTERNAL=0, SERVER=1, CLIENT=2, PRODUCER=3, CONSUMER=4）
   */
  start(spanName: string, kind?: number): ISpan;

  /** 创建 SERVER 类型 Span */
  startServer(spanName: string): ISpan;

  /** 创建 CLIENT 类型 Span */
  startClient(spanName: string): ISpan;
}

// ============================
// ILogBackend — Log 抽象
// ============================

/**
 * Log 后端接口。
 *
 * 实现：
 * - NoopLogBackend    — 空操作（开源默认）
 * - ConsoleLogBackend — stdout 输出（开发调试）
 * - OTelLogBackend    — OpenTelemetry Logs API（内部：智研 + ClickHouse 双写）
 */
export interface ILogBackend {
  /** 后端标识 */
  readonly type: string;

  /** INFO 级别日志 */
  info(eventName: string, attrs?: LogAttrs): void;

  /** WARN 级别日志 */
  warn(eventName: string, attrs?: LogAttrs): void;

  /** ERROR 级别日志 */
  error(eventName: string, attrs?: LogAttrs, error?: Error): void;

  /** DEBUG 级别日志 */
  debug?(eventName: string, attrs?: LogAttrs): void;
}

// ============================
// IMetricBackend — Metric 抽象
// ============================

/** 指标消息结构 */
export interface MetricMessage {
  /** 指标名 */
  metric: string;
  /** 实例 ID（同时作为 Kafka key） */
  instanceId: string;
  /** 原始值 */
  value: number;
  /** 事件发生时 Unix 秒（UTC），不传则自动取当前时间 */
  timestamp?: number;
  /** 来源服务 */
  source?: string;
  /**
   * 关联的 OTel Trace ID，用于 Metric → Trace 反查。
   * 由 metricProducer.send() 自动从当前 active span 注入，调用方通常不需要手动传入。
   * 存入 ClickHouse 后，在线评测服务可通过此字段定位到具体请求的完整 Trace。
   */
  traceId?: string;
}

/** Kafka Metric 配置 */
export interface MetricBackendConfig {
  /** Kafka Broker 列表 */
  brokers: string[];
  /** Topic 名称 (默认: "memory_monitor") */
  topic?: string;
  /** 是否启用 (默认: false) */
  enabled?: boolean;
}

/**
 * Metric 后端接口。
 *
 * 实现：
 * - NoopMetricBackend  — 空操作（开源默认）
 * - ConsoleMetricBackend — stdout 输出（开发调试）
 * - KafkaMetricBackend — Kafka Producer（内部：memory-monitor 消费 → Barad + ClickHouse）
 */
export interface IMetricBackend {
  /** 后端标识 */
  readonly type: string;

  /** 发送一条指标消息（同步，不阻塞） */
  send(msg: MetricMessage): void;

  /** 初始化后端（异步） */
  initialize(config: MetricBackendConfig): Promise<void>;

  /** 优雅关闭 */
  destroy(): Promise<void>;
}

// ============================
// ILLMTraceBackend — AI/LLM Trace 抽象
// ============================

/** Langfuse 配置 */
export interface LLMTraceConfig {
  /** 是否启用 */
  enabled: boolean;
  /** Langfuse 实例地址 */
  host?: string;
  /** Langfuse 公钥 */
  publicKey?: string;
  /** Langfuse 私钥 */
  secretKey?: string;
}

/**
 * LLM Trace 后端接口。
 *
 * 实现：
 * - NoopLLMTraceBackend     — 空操作（开源默认）
 * - ConsoleLLMTraceBackend  — stdout 输出（开发调试）
 * - LangfuseLLMTraceBackend — Langfuse（内部：过滤 AI Span 上报到 Langfuse）
 */
export interface ILLMTraceBackend {
  /** 后端标识 */
  readonly type: string;

  /**
   * 创建一个 SpanProcessor 实例。
   * 返回的 processor 会被注册到 OTel TracerProvider 中，
   * 用于过滤并上报 AI/LLM 相关 Span。
   */
  createSpanProcessor(): ISpanProcessor | null;

  /** 强制刷新待上报的 LLM Trace 数据 */
  flush(): Promise<void>;

  /** 优雅关闭 */
  shutdown(): Promise<void>;
}

// ============================
// ITraceMiddleware — HTTP Trace 中间件抽象
// ============================

/**
 * HTTP Trace 中间件接口。
 *
 * 实现：
 * - NoopTraceMiddleware — 透传（开源默认）
 * - ConsoleTraceMiddleware — stdout 输出（开发调试）
 * - OTelTraceMiddleware — OpenTelemetry（内部：创建 SERVER Span + Context 传播）
 */
export interface ITraceMiddleware {
  /** 后端标识 */
  readonly type: string;

  /**
   * 包装 HTTP 请求处理器，添加 Trace 埋点。
   * 从 traceparent 头恢复上游 Trace Context，创建 SERVER Span。
   */
  wrapWithTrace(
    req: http.IncomingMessage,
    res: http.ServerResponse,
    handler: () => Promise<void>,
  ): Promise<void>;

  /**
   * 创建子 Span（在业务处理器内部使用）。
   * 调用方需手动 span.end()。
   */
  startChildSpan(
    name: string,
    attrs?: Record<string, string | number | boolean>,
  ): ISpan;

  /**
   * 在 Span 上下文中执行函数，自动创建子 Span。
   */
  withSpan<T>(
    name: string,
    attrs: Record<string, string | number | boolean>,
    fn: (span: ISpan) => Promise<T>,
  ): Promise<T>;
}

// ============================
// ITracePropagation — Trace Context 传播抽象
// ============================

/**
 * Trace Context 跨异步边界传播接口。
 * 用于异步任务中序列化/反序列化 OTel Trace Context。
 */
export interface ITracePropagation {
  /**
   * 序列化当前 Trace Context 到 plain object。
   * 返回的 object 可以 spread 到 TaskPayload.data 中。
   */
  serializeTraceContext(): Record<string, string | number>;

  /**
   * 从 TaskPayload.data 反序列化恢复 Trace Context。
   * 返回 parentContext 和 parentSpanContext。
   */
  deserializeTraceContext(data?: Record<string, unknown>): {
    parentContext: unknown;
    parentSpanContext: { traceId: string; spanId: string; traceFlags: number; isRemote: boolean } | null;
  };
}

// ============================
// IObservabilityBackend — 聚合接口
// ============================

/**
 * 可观测性后端聚合接口 — 包含所有子后端。
 *
 * 通过工厂函数 createObservabilityBackend(config) 创建，
 * 全局单例暴露给各门面模块使用。
 */
export interface IObservabilityBackend {
  /** 后端类型标识 */
  readonly type: "noop" | "console" | "internal" | string;

  /** Trace 后端 */
  readonly trace: ITraceBackend;

  /** Log 后端 */
  readonly log: ILogBackend;

  /** Metric 后端 */
  readonly metric: IMetricBackend;

  /** LLM Trace 后端 */
  readonly llmTrace: ILLMTraceBackend;

  /** HTTP Trace 中间件 */
  readonly traceMiddleware: ITraceMiddleware;

  /** Trace Context 传播 */
  readonly tracePropagation: ITracePropagation;

  /** 初始化所有子后端 */
  initialize(config: ObservabilityConfig): Promise<void>;

  /** 优雅关闭所有子后端 */
  shutdown(): Promise<void>;
}

// ============================
// Configuration
// ============================

/** OTel 配置 */
export interface OTelConfig {
  /** 是否启用 */
  enabled: boolean;
  /** OTel Collector 端点 */
  endpoint?: string;
  /** 协议 (grpc | http | http/protobuf) */
  protocol?: "grpc" | "http" | "http/protobuf";
  /** 服务名 */
  serviceName?: string;
  /** 租户 ID */
  tenantId?: string;
}

/** ClickHouse 配置 */
export interface ClickHouseConfig {
  /** 是否启用 */
  enabled: boolean;
  /** ClickHouse HTTP 端点 */
  endpoint?: string;
  /** 用户名 */
  username?: string;
  /** 密码 */
  password?: string;
  /** 数据库名 */
  database?: string;
}

/**
 * 可观测性总配置。
 *
 * 后端类型说明：
 * - "noop"     — 空操作，零开销（默认，不配置时使用）
 * - "console"  — 输出到 stdout，用于开发调试
 * - "otlp"     — 标准 OTLP 协议后端（开源用户推荐）
 *                配置 otel.endpoint 即可将 Trace/Log/Metric 上报到任何支持 OTLP 的后端
 *                （如 ClickHouse、Jaeger、Grafana Tempo/Loki/Mimir、SigNoz、OTel Collector 等）
 * - "internal" — 内部私有模块（通过 git submodule 引入，智研 + Kafka + Langfuse）
 */
export interface ObservabilityConfig {
  /** 后端类型：noop | console | otlp | internal */
  type: "noop" | "console" | "otlp" | "internal" | string;

  /**
   * OTel 配置（otlp 和 internal 模式使用）。
   *
   * 开源用户使用 "otlp" 模式时，只需配置：
   *   otel: {
   *     enabled: true,
   *     endpoint: "http://localhost:4318",  // 你的 OTLP 后端地址
   *     serviceName: "my-memory-service",   // 可选，默认 "tdai-memory"
   *   }
   */
  otel?: OTelConfig;

  /** ClickHouse 配置（internal 模式使用） */
  clickhouse?: ClickHouseConfig;

  /** Kafka Metric 配置（internal 模式使用） */
  kafka?: MetricBackendConfig;

  /** Langfuse LLM Trace 配置（internal 模式使用） */
  langfuse?: LLMTraceConfig;
}

// ============================
// Logger Interface (for internal use)
// ============================

/** 可观测性模块内部使用的最小日志接口 */
export interface ObservabilityLogger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}
