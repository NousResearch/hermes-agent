/**
 * ClickHouse 直写 Exporter — 基于 @clickhouse/client 官方 SDK
 *
 * 性能最优方案，SDK 直连 ClickHouse HTTP 接口，无需 OTel Collector：
 * - ✅ gzip 压缩（减少 ~80% 网络流量）
 * - ✅ async_insert（ClickHouse 服务端合并小批量，减少 Part 数量）
 * - ✅ Keep-Alive 连接池（减少 TCP 握手开销）
 * - ✅ 内部批量缓冲 + 定时 flush
 * - ✅ 失败静默，不阻塞业务
 *
 * 数据流（无 Collector，直连 ClickHouse）：
 *   SDK TracerProvider ──→ [智研 OTLP Exporter] + [ClickHouse Exporter (本模块)]
 *   SDK LoggerProvider  ──→ [智研 OTLP Exporter] + [ClickHouse Exporter (本模块)]
 *   SDK MeterProvider   ──→ [智研监控宝 Bridge]  + [ClickHouse Exporter (本模块)]
 *
 * 使用方式：
 *   const chExporter = new ClickHouseDirectExporter({ endpoint: "http://10.0.1.100:8123", ... });
 *   // 由 otel-sdk-init.ts 在初始化 SDK 时注册为额外的 Exporter
 *
 * 环境变量：
 *   - CLICKHOUSE_HTTP_ENDPOINT : ClickHouse HTTP 接口地址 (默认: http://localhost:8123)
 *   - CLICKHOUSE_USER          : 用户名 (默认: "default")
 *   - CLICKHOUSE_PASSWORD      : 密码
 *   - CLICKHOUSE_DATABASE      : 数据库名 (默认: "tdai_eval")
 *   - CLICKHOUSE_ENABLED       : "true" 启用 (默认: "false")
 */

import { createClient, type ClickHouseClient } from "@clickhouse/client";

export interface ClickHouseExporterOptions {
  /** ClickHouse HTTP 接口地址 (端口 8123). 默认: env CLICKHOUSE_HTTP_ENDPOINT */
  endpoint?: string;
  /** 用户名. 默认: env CLICKHOUSE_USER 或 "default" */
  username?: string;
  /** 密码. 默认: env CLICKHOUSE_PASSWORD */
  password?: string;
  /** 数据库名. 默认: env CLICKHOUSE_DATABASE 或 "tdai_eval" */
  database?: string;
  /** Trace 表名. 默认: "otel_traces" */
  tracesTable?: string;
  /** Log 表名. 默认: "otel_logs" */
  logsTable?: string;
  /** Metrics Gauge 表名. 默认: "otel_metrics_gauge" */
  metricsGaugeTable?: string;
  /** Metrics Sum 表名. 默认: "otel_metrics_sum" */
  metricsSumTable?: string;
  /** Metrics Histogram 表名. 默认: "otel_metrics_histogram" */
  metricsHistogramTable?: string;
  /** 批量发送大小. 默认: 100 */
  batchSize?: number;
  /** 定时 flush 间隔 ms. 默认: 5000 */
  flushIntervalMs?: number;
  /** 请求超时 ms. 默认: 30000 */
  timeoutMs?: number;
  /** 是否启用 debug 日志 */
  debug?: boolean;
  /** 服务名（写入 ServiceName 列） */
  serviceName?: string;
  /** 最大并发 insert 请求数. 默认: 5 */
  maxConcurrentInserts?: number;
  /** 是否启用请求体 gzip 压缩. 默认: true */
  compression?: boolean;
}

// ── 内部类型 ──

interface TraceRow {
  Timestamp: string;
  TraceId: string;
  SpanId: string;
  ParentSpanId: string;
  TraceState: string;
  SpanName: string;
  SpanKind: string;
  ServiceName: string;
  Duration: number;
  StatusCode: string;
  StatusMessage: string;
  SpanAttributes: Record<string, string>;
  ResourceAttributes: Record<string, string>;
  "Events.Timestamp": string[];
  "Events.Name": string[];
  "Events.Attributes": Record<string, string>[];
  "Links.TraceId": string[];
  "Links.SpanId": string[];
  "Links.TraceState": string[];
  "Links.Attributes": Record<string, string>[];
}

interface LogRow {
  Timestamp: string;
  ObservedTimestamp: string;
  TraceId: string;
  SpanId: string;
  TraceFlags: number;
  SeverityText: string;
  SeverityNumber: number;
  ServiceName: string;
  Body: string;
  LogAttributes: Record<string, string>;
  ResourceAttributes: Record<string, string>;
  ScopeName: string;
  ScopeVersion: string;
}

interface MetricGaugeRow {
  TimeUnix: string;
  MetricName: string;
  ServiceName: string;
  Value: number;
  Attributes: Record<string, string>;
  ResourceAttributes: Record<string, string>;
  ScopeName: string;
  ScopeVersion: string;
}

interface MetricSumRow {
  TimeUnix: string;
  MetricName: string;
  ServiceName: string;
  Value: number;
  IsMonotonic: boolean;
  AggregationTemporality: string;
  Attributes: Record<string, string>;
  ResourceAttributes: Record<string, string>;
  ScopeName: string;
  ScopeVersion: string;
}

interface MetricHistogramRow {
  TimeUnix: string;
  MetricName: string;
  ServiceName: string;
  Count: number;
  Sum: number;
  Min: number;
  Max: number;
  BucketCounts: number[];
  ExplicitBounds: number[];
  Attributes: Record<string, string>;
  ResourceAttributes: Record<string, string>;
  ScopeName: string;
  ScopeVersion: string;
}

/**
 * ClickHouse 直写 Exporter — 基于 @clickhouse/client 官方 SDK.
 *
 * 暴露简单的 exportSpan / exportLog / exportMetric 方法，
 * 由 otel-sdk-init.ts 在创建 OTel 各 Provider 时挂载。
 */
export class ClickHouseDirectExporter {
  private readonly client: ClickHouseClient;
  private readonly database: string;
  private readonly tracesTable: string;
  private readonly logsTable: string;
  private readonly metricsGaugeTable: string;
  private readonly metricsSumTable: string;
  private readonly metricsHistogramTable: string;
  private readonly batchSize: number;
  private readonly flushIntervalMs: number;
  private readonly debug: boolean;
  private readonly serviceName: string;
  private readonly maxConcurrentInserts: number;

  // 内部缓冲
  private traceBuffer: TraceRow[] = [];
  private logBuffer: LogRow[] = [];
  private metricGaugeBuffer: MetricGaugeRow[] = [];
  private metricSumBuffer: MetricSumRow[] = [];
  private metricHistogramBuffer: MetricHistogramRow[] = [];

  // 并发控制
  private activeInserts = 0;

  private flushTimer: ReturnType<typeof setInterval> | undefined;
  private _shutdown = false;

  constructor(options: ClickHouseExporterOptions = {}) {
    const endpoint = options.endpoint
      ?? process.env.CLICKHOUSE_HTTP_ENDPOINT
      ?? "http://localhost:8123";
    const username = options.username
      ?? process.env.CLICKHOUSE_USER
      ?? "default";
    const password = options.password
      ?? process.env.CLICKHOUSE_PASSWORD
      ?? "";
    this.database = options.database
      ?? process.env.CLICKHOUSE_DATABASE
      ?? "tdai_eval";
    const timeoutMs = options.timeoutMs ?? 30_000;
    const compression = options.compression ?? true;

    // 创建 @clickhouse/client 实例 — 性能最优配置
    this.client = createClient({
      url: endpoint,
      username,
      password,
      database: this.database,
      // gzip 压缩：减少 ~80% 网络流量
      compression: {
        request: compression,
      },
      // ClickHouse 服务端异步插入 + Keep-Alive
      clickhouse_settings: {
        async_insert: 1,              // 异步插入，CK 服务端合并小批量
        wait_for_async_insert: 1,     // 等待写入完成确认
      },
      // 请求超时
      request_timeout: timeoutMs,
      // Keep-Alive 连接池配置
      keep_alive: {
        enabled: true,
      },
    });

    this.tracesTable = options.tracesTable ?? "otel_traces";
    this.logsTable = options.logsTable ?? "otel_logs";
    this.metricsGaugeTable = options.metricsGaugeTable ?? "otel_metrics_gauge";
    this.metricsSumTable = options.metricsSumTable ?? "otel_metrics_sum";
    this.metricsHistogramTable = options.metricsHistogramTable ?? "otel_metrics_histogram";
    this.batchSize = options.batchSize ?? 100;
    this.flushIntervalMs = options.flushIntervalMs ?? 5000;
    this.debug = options.debug ?? (process.env.OTEL_LOG_LEVEL === "DEBUG");
    this.serviceName = options.serviceName
      ?? process.env.OTEL_SERVICE_NAME
      ?? "core";
    this.maxConcurrentInserts = options.maxConcurrentInserts ?? 5;

    // 定时 flush
    this.flushTimer = setInterval(() => {
      void this.flush();
    }, this.flushIntervalMs);
    if (this.flushTimer.unref) {
      this.flushTimer.unref();
    }

    if (this.debug) {
      console.info(
        `[clickhouse-exporter] Initialized (SDK mode) | endpoint=${endpoint} | database=${this.database} | compression=${compression}`
      );
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Public API: 导出数据
  // ════════════════════════════════════════════════════════════════════════════

  /**
   * 导出一个 Span (Trace) 到 ClickHouse.
   */
  exportSpan(span: {
    traceId: string;
    spanId: string;
    parentSpanId?: string;
    traceState?: string;
    name: string;
    kind: number; // OTel SpanKind enum
    startTime: [number, number] | number; // hrtime or epoch nanos
    endTime: [number, number] | number;
    status?: { code: number; message?: string };
    attributes?: Record<string, unknown>;
    resource?: Record<string, unknown>;
    events?: Array<{
      name: string;
      time: [number, number] | number;
      attributes?: Record<string, unknown>;
    }>;
    links?: Array<{
      traceId: string;
      spanId: string;
      traceState?: string;
      attributes?: Record<string, unknown>;
    }>;
  }): void {
    if (this._shutdown) return;

    try {
      if (this.debug) {
        console.debug(`[clickhouse-exporter] exportSpan: ${span.name} (traceId=${span.traceId?.substring(0,8)}...)`);
      }
      const startNanos = hrtimeToNanos(span.startTime);
      const endNanos = hrtimeToNanos(span.endTime);
      const durationNanos = endNanos - startNanos;

      const row: TraceRow = {
        Timestamp: nanosToClickHouseTimestamp(startNanos),
        TraceId: span.traceId,
        SpanId: span.spanId,
        ParentSpanId: span.parentSpanId ?? "",
        TraceState: span.traceState ?? "",
        SpanName: span.name,
        SpanKind: spanKindToString(span.kind),
        ServiceName: this.serviceName,
        Duration: Number(durationNanos),
        StatusCode: statusCodeToString(span.status?.code ?? 0),
        StatusMessage: span.status?.message ?? "",
        SpanAttributes: flattenAttributes(span.attributes),
        ResourceAttributes: flattenAttributes(span.resource),
        "Events.Timestamp": (span.events ?? []).map(e => nanosToClickHouseTimestamp(hrtimeToNanos(e.time))),
        "Events.Name": (span.events ?? []).map(e => e.name),
        "Events.Attributes": (span.events ?? []).map(e => flattenAttributes(e.attributes)),
        "Links.TraceId": (span.links ?? []).map(l => l.traceId),
        "Links.SpanId": (span.links ?? []).map(l => l.spanId),
        "Links.TraceState": (span.links ?? []).map(l => l.traceState ?? ""),
        "Links.Attributes": (span.links ?? []).map(l => flattenAttributes(l.attributes)),
      };

      this.traceBuffer.push(row);
      if (this.traceBuffer.length >= this.batchSize) {
        void this.flushTraces();
      }
    } catch {
      // Never block business logic
    }
  }

  /**
   * 导出一条 Log 到 ClickHouse.
   */
  exportLog(log: {
    timestamp?: number; // epoch nanos
    observedTimestamp?: number;
    traceId?: string;
    spanId?: string;
    traceFlags?: number;
    severityText?: string;
    severityNumber?: number;
    body?: string;
    attributes?: Record<string, unknown>;
    resource?: Record<string, unknown>;
    scopeName?: string;
    scopeVersion?: string;
  }): void {
    if (this._shutdown) return;

    try {
      const now = Date.now() * 1_000_000; // current time in nanos
      const row: LogRow = {
        Timestamp: nanosToClickHouseTimestamp(log.timestamp ?? now),
        ObservedTimestamp: nanosToClickHouseTimestamp(log.observedTimestamp ?? now),
        TraceId: log.traceId ?? "",
        SpanId: log.spanId ?? "",
        TraceFlags: log.traceFlags ?? 0,
        SeverityText: log.severityText ?? "INFO",
        SeverityNumber: log.severityNumber ?? 9,
        ServiceName: this.serviceName,
        Body: log.body ?? "",
        LogAttributes: flattenAttributes(log.attributes),
        ResourceAttributes: flattenAttributes(log.resource),
        ScopeName: log.scopeName ?? "",
        ScopeVersion: log.scopeVersion ?? "",
      };

      this.logBuffer.push(row);
      if (this.logBuffer.length >= this.batchSize) {
        void this.flushLogs();
      }
    } catch {
      // Never block business logic
    }
  }

  /**
   * 导出 Gauge 指标到 ClickHouse.
   */
  exportGauge(metric: {
    name: string;
    value: number;
    timestamp?: number; // epoch nanos
    attributes?: Record<string, string>;
    resource?: Record<string, string>;
  }): void {
    if (this._shutdown) return;

    try {
      const now = Date.now() * 1_000_000;
      const row: MetricGaugeRow = {
        TimeUnix: nanosToClickHouseTimestamp(metric.timestamp ?? now),
        MetricName: metric.name,
        ServiceName: this.serviceName,
        Value: metric.value,
        Attributes: metric.attributes ?? {},
        ResourceAttributes: metric.resource ?? {},
        ScopeName: "memory-tdai",
        ScopeVersion: "",
      };

      this.metricGaugeBuffer.push(row);
      if (this.metricGaugeBuffer.length >= this.batchSize) {
        void this.flushMetricsGauge();
      }
    } catch {
      // Never block business logic
    }
  }

  /**
   * 导出 Counter (Sum) 指标到 ClickHouse.
   */
  exportSum(metric: {
    name: string;
    value: number;
    isMonotonic?: boolean;
    timestamp?: number;
    attributes?: Record<string, string>;
    resource?: Record<string, string>;
  }): void {
    if (this._shutdown) return;

    try {
      const now = Date.now() * 1_000_000;
      const row: MetricSumRow = {
        TimeUnix: nanosToClickHouseTimestamp(metric.timestamp ?? now),
        MetricName: metric.name,
        ServiceName: this.serviceName,
        Value: metric.value,
        IsMonotonic: metric.isMonotonic ?? true,
        AggregationTemporality: "Cumulative",
        Attributes: metric.attributes ?? {},
        ResourceAttributes: metric.resource ?? {},
        ScopeName: "memory-tdai",
        ScopeVersion: "",
      };

      this.metricSumBuffer.push(row);
      if (this.metricSumBuffer.length >= this.batchSize) {
        void this.flushMetricsSum();
      }
    } catch {
      // Never block business logic
    }
  }

  /**
   * 导出 Histogram 指标到 ClickHouse.
   */
  exportHistogram(metric: {
    name: string;
    count: number;
    sum: number;
    min?: number;
    max?: number;
    bucketCounts?: number[];
    explicitBounds?: number[];
    timestamp?: number;
    attributes?: Record<string, string>;
    resource?: Record<string, string>;
  }): void {
    if (this._shutdown) return;

    try {
      const now = Date.now() * 1_000_000;
      const row: MetricHistogramRow = {
        TimeUnix: nanosToClickHouseTimestamp(metric.timestamp ?? now),
        MetricName: metric.name,
        ServiceName: this.serviceName,
        Count: metric.count,
        Sum: metric.sum,
        Min: metric.min ?? 0,
        Max: metric.max ?? 0,
        BucketCounts: metric.bucketCounts ?? [],
        ExplicitBounds: metric.explicitBounds ?? [],
        Attributes: metric.attributes ?? {},
        ResourceAttributes: metric.resource ?? {},
        ScopeName: "memory-tdai",
        ScopeVersion: "",
      };

      this.metricHistogramBuffer.push(row);
      if (this.metricHistogramBuffer.length >= this.batchSize) {
        void this.flushMetricsHistogram();
      }
    } catch {
      // Never block business logic
    }
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Flush & Shutdown
  // ════════════════════════════════════════════════════════════════════════════

  /**
   * Flush 所有缓冲数据到 ClickHouse.
   */
  async flush(): Promise<void> {
    await Promise.allSettled([
      this.flushTraces(),
      this.flushLogs(),
      this.flushMetricsGauge(),
      this.flushMetricsSum(),
      this.flushMetricsHistogram(),
    ]);
  }

  /**
   * 优雅关闭：flush 剩余数据，关闭连接池.
   */
  async shutdown(): Promise<void> {
    this._shutdown = true;
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
      this.flushTimer = undefined;
    }
    await this.flush();
    await this.client.close();
    if (this.debug) {
      console.info("[clickhouse-exporter] Shutdown complete");
    }
  }

  /**
   * 是否已关闭.
   */
  get isShutdown(): boolean {
    return this._shutdown;
  }

  // ════════════════════════════════════════════════════════════════════════════
  // Internal: 分表 flush
  // ════════════════════════════════════════════════════════════════════════════

  private async flushTraces(): Promise<void> {
    if (this.traceBuffer.length === 0) return;
    const rows = this.traceBuffer.splice(0);
    await this.insertRows(this.tracesTable, rows);
  }

  private async flushLogs(): Promise<void> {
    if (this.logBuffer.length === 0) return;
    const rows = this.logBuffer.splice(0);
    await this.insertRows(this.logsTable, rows);
  }

  private async flushMetricsGauge(): Promise<void> {
    if (this.metricGaugeBuffer.length === 0) return;
    const rows = this.metricGaugeBuffer.splice(0);
    await this.insertRows(this.metricsGaugeTable, rows);
  }

  private async flushMetricsSum(): Promise<void> {
    if (this.metricSumBuffer.length === 0) return;
    const rows = this.metricSumBuffer.splice(0);
    await this.insertRows(this.metricsSumTable, rows);
  }

  private async flushMetricsHistogram(): Promise<void> {
    if (this.metricHistogramBuffer.length === 0) return;
    const rows = this.metricHistogramBuffer.splice(0);
    await this.insertRows(this.metricsHistogramTable, rows);
  }

  /**
   * 通过 @clickhouse/client SDK 批量插入数据.
   * 自动 gzip 压缩 + async_insert + Keep-Alive 连接池.
   */
  private async insertRows(table: string, rows: unknown[]): Promise<void> {
    if (rows.length === 0) return;

    // 并发控制：避免过多并发请求打爆 ClickHouse
    if (this.activeInserts >= this.maxConcurrentInserts) {
      // 放回缓冲区
      this.requeue(table, rows);
      return;
    }

    this.activeInserts++;
    try {
      await this.client.insert({
        table,
        values: rows,
        format: "JSONEachRow",
      });

      if (this.debug) {
        console.debug(`[clickhouse-exporter] INSERT ${table}: ${rows.length} rows OK`);
      }
    } catch (err) {
      if (this.debug) {
        console.warn(
          `[clickhouse-exporter] INSERT ${table} error: ${err instanceof Error ? err.message : String(err)}`
        );
      }
      // 失败的数据放回缓冲区
      this.requeue(table, rows);
    } finally {
      this.activeInserts--;
    }
  }

  /**
   * 失败时有限重新入队（最多保留 2x batchSize 避免 OOM）.
   */
  private requeue(table: string, rows: unknown[]): void {
    const maxBuffer = this.batchSize * 2;

    if (table === this.tracesTable && this.traceBuffer.length < maxBuffer) {
      this.traceBuffer.unshift(...(rows as TraceRow[]));
    } else if (table === this.logsTable && this.logBuffer.length < maxBuffer) {
      this.logBuffer.unshift(...(rows as LogRow[]));
    } else if (table === this.metricsGaugeTable && this.metricGaugeBuffer.length < maxBuffer) {
      this.metricGaugeBuffer.unshift(...(rows as MetricGaugeRow[]));
    } else if (table === this.metricsSumTable && this.metricSumBuffer.length < maxBuffer) {
      this.metricSumBuffer.unshift(...(rows as MetricSumRow[]));
    } else if (table === this.metricsHistogramTable && this.metricHistogramBuffer.length < maxBuffer) {
      this.metricHistogramBuffer.unshift(...(rows as MetricHistogramRow[]));
    }
    // else: drop data (buffer full, avoid OOM)
  }
}

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

/**
 * 将 hrtime [seconds, nanoseconds] 或 epoch nanos 转为纳秒数.
 */
function hrtimeToNanos(time: [number, number] | number): bigint {
  if (typeof time === "number") {
    return BigInt(time);
  }
  return BigInt(time[0]) * 1_000_000_000n + BigInt(time[1]);
}

/**
 * 将纳秒时间戳转为 ClickHouse DateTime64(9) 格式.
 * 格式: "YYYY-MM-DD HH:mm:ss.NNNNNNNNN"
 */
function nanosToClickHouseTimestamp(nanos: bigint | number): string {
  const ns = BigInt(nanos);
  const ms = Number(ns / 1_000_000n);
  const remainingNanos = Number(ns % 1_000_000_000n);

  const date = new Date(ms);
  const yyyy = date.getUTCFullYear();
  const MM = String(date.getUTCMonth() + 1).padStart(2, "0");
  const dd = String(date.getUTCDate()).padStart(2, "0");
  const HH = String(date.getUTCHours()).padStart(2, "0");
  const mm = String(date.getUTCMinutes()).padStart(2, "0");
  const ss = String(date.getUTCSeconds()).padStart(2, "0");
  const nanoStr = String(remainingNanos).padStart(9, "0");

  return `${yyyy}-${MM}-${dd} ${HH}:${mm}:${ss}.${nanoStr}`;
}

/**
 * SpanKind 数字 → 字符串.
 */
function spanKindToString(kind: number): string {
  switch (kind) {
    case 0: return "INTERNAL";
    case 1: return "SERVER";
    case 2: return "CLIENT";
    case 3: return "PRODUCER";
    case 4: return "CONSUMER";
    default: return "INTERNAL";
  }
}

/**
 * StatusCode 数字 → 字符串.
 */
function statusCodeToString(code: number): string {
  switch (code) {
    case 0: return "UNSET";
    case 1: return "OK";
    case 2: return "ERROR";
    default: return "UNSET";
  }
}

/**
 * 将 OTel Attributes (可能是嵌套对象) 展平为 Map(String, String).
 */
function flattenAttributes(attrs?: Record<string, unknown>): Record<string, string> {
  if (!attrs) return {};
  const result: Record<string, string> = {};
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined) continue;
    if (typeof value === "string") {
      result[key] = value;
    } else if (typeof value === "number" || typeof value === "boolean") {
      result[key] = String(value);
    } else if (Array.isArray(value)) {
      result[key] = JSON.stringify(value);
    } else if (typeof value === "object") {
      result[key] = JSON.stringify(value);
    }
  }
  return result;
}

// ════════════════════════════════════════════════════════════════════════════
// OTel SpanExporter 适配器 — 让 ClickHouseDirectExporter 可以直接注册到
// OTel TracerProvider 作为标准 SpanExporter
// ════════════════════════════════════════════════════════════════════════════

/**
 * 适配 OTel SpanExporter 接口.
 * 用法：
 *   const chExporter = new ClickHouseDirectExporter({...});
 *   const spanExporter = new ClickHouseSpanExporter(chExporter);
 *   // 传给 NodeSDK 的 traceExporter 或作为额外 SpanProcessor
 */
export class ClickHouseSpanExporter {
  private ch: ClickHouseDirectExporter;

  constructor(chExporter: ClickHouseDirectExporter) {
    this.ch = chExporter;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export(spans: any[], resultCallback: (result: { code: number }) => void): void {
    try {
      if (process.env.OTEL_LOG_LEVEL === "DEBUG") {
        console.debug(`[clickhouse-exporter] export() called with ${spans.length} span(s): ${spans.map((s: any) => s.name).join(', ')}`);
      }
      for (const span of spans) {
        this.ch.exportSpan({
          traceId: span.spanContext?.().traceId ?? span._spanContext?.traceId ?? "",
          spanId: span.spanContext?.().spanId ?? span._spanContext?.spanId ?? "",
          parentSpanId: span.parentSpanId ?? "",
          traceState: span.spanContext?.().traceState?.serialize?.() ?? "",
          name: span.name ?? "",
          kind: span.kind ?? 0,
          startTime: span.startTime ?? span._startTime ?? Date.now() * 1_000_000,
          endTime: span.endTime ?? span._endTime ?? Date.now() * 1_000_000,
          status: span.status,
          attributes: span.attributes ?? {},
          resource: span.resource?.attributes ?? {},
          events: (span.events ?? []).map((e: { name: string; time: unknown; attributes?: Record<string, unknown> }) => ({
            name: e.name,
            time: e.time as [number, number] | number,
            attributes: e.attributes,
          })),
          links: (span.links ?? []).map((l: { context: { traceId: string; spanId: string; traceState?: { serialize?: () => string } }; attributes?: Record<string, unknown> }) => ({
            traceId: l.context?.traceId ?? "",
            spanId: l.context?.spanId ?? "",
            traceState: l.context?.traceState?.serialize?.() ?? "",
            attributes: l.attributes,
          })),
        });
      }
      resultCallback({ code: 0 }); // SUCCESS
    } catch {
      resultCallback({ code: 1 }); // FAILED
    }
  }

  async shutdown(): Promise<void> {
    await this.ch.shutdown();
  }

  async forceFlush(): Promise<void> {
    await this.ch.flush();
  }
}

/**
 * 适配 OTel LogRecordExporter 接口.
 */
export class ClickHouseLogExporter {
  private ch: ClickHouseDirectExporter;

  constructor(chExporter: ClickHouseDirectExporter) {
    this.ch = chExporter;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  export(logRecords: any[], resultCallback: (result: { code: number }) => void): void {
    try {
      for (const record of logRecords) {
        this.ch.exportLog({
          timestamp: hrtimeToNanosExport(record.hrTime ?? record._hrTime),
          observedTimestamp: hrtimeToNanosExport(record.observedHrTime ?? record.hrTime ?? record._hrTime),
          traceId: record.spanContext?.traceId ?? "",
          spanId: record.spanContext?.spanId ?? "",
          traceFlags: record.spanContext?.traceFlags ?? 0,
          severityText: record.severityText ?? "",
          severityNumber: record.severityNumber ?? 0,
          body: typeof record.body === "string" ? record.body : JSON.stringify(record.body ?? ""),
          attributes: record.attributes ?? {},
          resource: record.resource?.attributes ?? {},
          scopeName: record.instrumentationScope?.name ?? "",
          scopeVersion: record.instrumentationScope?.version ?? "",
        });
      }
      resultCallback({ code: 0 });
    } catch {
      resultCallback({ code: 1 });
    }
  }

  async shutdown(): Promise<void> {
    await this.ch.shutdown();
  }

  async forceFlush(): Promise<void> {
    await this.ch.flush();
  }
}

function hrtimeToNanosExport(time: [number, number] | number | undefined): number | undefined {
  if (time === undefined) return undefined;
  if (typeof time === "number") return time;
  return Number(BigInt(time[0]) * 1_000_000_000n + BigInt(time[1]));
}
