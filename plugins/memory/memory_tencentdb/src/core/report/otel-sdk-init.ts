/**
 * OTel SDK 初始化模块 — core 服务
 *
 * 基于参考代码（trace 分支 otel-sdk-init.ts）适配 core 服务。
 * 负责初始化 OpenTelemetry NodeSDK，支持 Trace/Metrics/Logs 三信号。
 *
 * 环境变量配置（OTel 标准）：
 * - OTEL_EXPORTER_OTLP_ENDPOINT    : Collector endpoint (默认: http://localhost:4317)
 * - OTEL_EXPORTER_OTLP_PROTOCOL    : "grpc" | "http/protobuf" (默认: "grpc")
 * - OTEL_EXPORTER_OTLP_HEADERS     : 逗号分隔的 key=value 对，用于鉴权
 * - OTEL_SERVICE_NAME               : 服务名 (默认: "core")
 * - OTEL_RESOURCE_ATTRIBUTES        : 额外 resource 属性（智研需要 tps.tenant.id）
 *
 * 自定义环境变量：
 * - TDAI_OTEL_ENABLED              : "true" 启用 OTel SDK (默认: "false")
 * - TDAI_INSTANCE_ID               : 实例标识
 * - TDAI_METRICS_MODE              : "otlp" | "none" (默认: "none")
 * - CLICKHOUSE_ENABLED             : "true" 启用 ClickHouse 双写
 */

// 防御性加载 @opentelemetry/api — 即使包缺失也不影响启动
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let diag: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let DiagConsoleLogger: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let DiagLogLevel: any;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let trace: any;
let _otelApiAvailable = false;

try {
  const api = await import("@opentelemetry/api");
  diag = api.diag;
  DiagConsoleLogger = api.DiagConsoleLogger;
  DiagLogLevel = api.DiagLogLevel;
  trace = api.trace;
  _otelApiAvailable = true;
} catch {
  // @opentelemetry/api 不可用，OTel SDK 初始化将被跳过
  console.warn("[core][otel] @opentelemetry/api not available, OTel SDK disabled.");
}

export interface OTelSDKInitOptions {
  serviceName?: string;
  serviceVersion?: string;
  instanceId?: string;
  endpoint?: string;
  protocol?: "grpc" | "http/protobuf";
  /** 智研租户 ID，会设置为 Resource Attribute "tps.tenant.id"（智研 APM 认证必需） */
  tenantId?: string;
  headers?: Record<string, string>;
  debug?: boolean;
  logExportIntervalMs?: number;
  clickhouse?: boolean | {
    endpoint?: string;
    username?: string;
    password?: string;
    database?: string;
  };
  /** Langfuse LLM trace 上报配置（只转发 ai 和 gen_ai 前缀的 span） */
  langfuse?: boolean | {
    host: string;
    publicKey: string;
    secretKey: string;
  };
}

let _sdkInstance: { shutdown: () => Promise<void> } | undefined;
let _initialized = false;

/**
 * 初始化 OpenTelemetry SDK。
 * 安全地多次调用 — 后续调用为 no-op。
 * 如果 SDK 包未安装，记录警告并返回 false。
 */
export async function initOTelSDK(options: OTelSDKInitOptions = {}): Promise<boolean> {
  if (_initialized) return true;

  // 当配置了 endpoint 或 Langfuse 时启用 SDK
  const hasLangfuse = typeof options.langfuse === "object" && options.langfuse && options.langfuse.host;
  const enabled = options.endpoint
    ? true
    : hasLangfuse
      ? true
      : process.env.TDAI_OTEL_ENABLED === "true";
  if (!enabled) return false;

  // @opentelemetry/api 不可用时直接返回
  if (!_otelApiAvailable) {
    console.warn("[core][otel] @opentelemetry/api not available, skipping SDK init.");
    return false;
  }

  if (options.debug || process.env.OTEL_LOG_LEVEL === "DEBUG") {
    diag.setLogger(new DiagConsoleLogger(), DiagLogLevel.DEBUG);
  }

  try {
    const [
      { NodeSDK },
      resourcesModule,
      { ATTR_SERVICE_NAME, ATTR_SERVICE_VERSION, ATTR_SERVICE_INSTANCE_ID },
      { OTLPTraceExporter: GrpcTraceExporter },
      { OTLPTraceExporter: HttpTraceExporter },
      { OTLPLogExporter: GrpcLogExporter },
      { OTLPLogExporter: HttpLogExporter },
      { LoggerProvider, BatchLogRecordProcessor },
      { AsyncLocalStorageContextManager },
    ] = await Promise.all([
      import("@opentelemetry/sdk-node"),
      import("@opentelemetry/resources"),
      import("@opentelemetry/semantic-conventions"),
      import("@opentelemetry/exporter-trace-otlp-grpc"),
      import("@opentelemetry/exporter-trace-otlp-http"),
      import("@opentelemetry/exporter-logs-otlp-grpc"),
      import("@opentelemetry/exporter-logs-otlp-http"),
      import("@opentelemetry/sdk-logs"),
      import("@opentelemetry/context-async-hooks"),
    ]);

    // 兼容新旧版本 @opentelemetry/resources
    // 新版使用 resourceFromAttributes()，旧版使用 new Resource()
    const createResource = (attrs: Record<string, string>) => {
      if ("resourceFromAttributes" in resourcesModule) {
        return (resourcesModule as { resourceFromAttributes: (a: Record<string, string>) => unknown }).resourceFromAttributes(attrs);
      }
      // 旧版 fallback
      const ResourceClass = (resourcesModule as { Resource: new (a: Record<string, string>) => unknown }).Resource;
      return new ResourceClass(attrs);
    };

    const logsApiModule = "@opentelemetry/api-logs";
    const { logs } = await import(logsApiModule);

    // 解析配置
    const endpoint = options.endpoint
      ?? process.env.OTEL_EXPORTER_OTLP_ENDPOINT
      ?? "";

    // 是否有主 OTel collector（区别于仅 Langfuse 场景）
    const hasMainOtel = Boolean(endpoint);

    const protocol = options.protocol
      ?? (process.env.OTEL_EXPORTER_OTLP_PROTOCOL as "grpc" | "http/protobuf")
      ?? "grpc";

    const headers = options.headers ?? parseHeadersFromEnv();
    const serviceName = options.serviceName ?? process.env.OTEL_SERVICE_NAME ?? "core";
    const serviceVersion = options.serviceVersion ?? "unknown";
    const os = await import("node:os");
    const instanceId = options.instanceId ?? process.env.TDAI_INSTANCE_ID ?? process.env.HOSTNAME ?? os.hostname() ?? "unknown";

    // 构建 Resource（智研 APM 需要 tps.tenant.id 作为 Resource Attribute 认证）
    const extraResourceAttrs = parseResourceAttributesFromEnv();
    const tenantId = options.tenantId ?? extraResourceAttrs["tps.tenant.id"] ?? "";
    const resourceAttrs: Record<string, string> = {
      [ATTR_SERVICE_NAME]: serviceName,
      [ATTR_SERVICE_VERSION]: serviceVersion,
      [ATTR_SERVICE_INSTANCE_ID]: instanceId,
      ...extraResourceAttrs,
    };
    // 确保 tps.tenant.id 被设置为 Resource Attribute（智研 APM 认证方式）
    if (tenantId) {
      resourceAttrs["tps.tenant.id"] = tenantId;
    }
    const resource = createResource(resourceAttrs);

    // Trace Exporter（仅当有主 OTel endpoint 时创建）
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let traceExporter: any = null;
    if (hasMainOtel) {
      traceExporter = protocol === "grpc"
        ? new GrpcTraceExporter({ url: endpoint, headers })
        : new HttpTraceExporter({ url: `${endpoint}/v1/traces`, headers });
    }

    // 注意：Metric 不走 OTLP，通过 Kafka 上报。

    // Log Exporter（仅当有主 OTel endpoint 时创建）
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    let logExporter: any = null;
    if (hasMainOtel) {
      logExporter = protocol === "grpc"
        ? new GrpcLogExporter({ url: endpoint, headers })
        : new HttpLogExporter({ url: `${endpoint}/v1/logs`, headers });
    }

    // 收集所有 Log Processors（新版 LoggerProvider 需要在构造时传入）
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const logProcessors: any[] = [];
    if (logExporter) {
      logProcessors.push(
        new BatchLogRecordProcessor(logExporter, {
          maxExportBatchSize: 512,
          scheduledDelayMillis: options.logExportIntervalMs ?? 5_000,
        }),
      );
    }

    // ClickHouse 双写（可选）
    // options.clickhouse 为对象时视为已启用，为 true 时也视为启用，
    // 为 false/undefined 时回退检查环境变量。
    const clickhouseEnabled = (typeof options.clickhouse === "object" && options.clickhouse !== null)
      || options.clickhouse === true
      || (options.clickhouse !== false && process.env.CLICKHOUSE_ENABLED === "true");

    let clickhouseShutdown: (() => Promise<void>) | undefined;

    if (clickhouseEnabled) {
      try {
        const { ClickHouseDirectExporter, ClickHouseSpanExporter, ClickHouseLogExporter } =
          await import("./clickhouse-exporter.js");

        const chOpts = typeof options.clickhouse === "object" ? options.clickhouse : {};
        const chExporter = new ClickHouseDirectExporter({
          endpoint: chOpts.endpoint,
          username: chOpts.username,
          password: chOpts.password,
          database: chOpts.database,
          serviceName,
          debug: options.debug,
        });

        // 添加 ClickHouse Log Processor
        const chLogExporter = new ClickHouseLogExporter(chExporter);
        logProcessors.push(
          new BatchLogRecordProcessor(chLogExporter as unknown as InstanceType<typeof GrpcLogExporter>, {
            maxExportBatchSize: 512,
            scheduledDelayMillis: options.logExportIntervalMs ?? 5_000,
          }),
        );

        const _chSpanExporter = new ClickHouseSpanExporter(chExporter);
        (globalThis as Record<string, unknown>).__chSpanExporter = _chSpanExporter;

        clickhouseShutdown = async () => {
          await chExporter.shutdown();
        };

        console.info(`[core][otel] ClickHouse direct-write enabled ✓`);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.warn(`[core][otel] ClickHouse exporter init failed: ${msg}. Continuing without ClickHouse.`);
      }
    }

    // 创建 LoggerProvider（新版 API：构造时传入 resource + processors）
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const loggerProvider = new LoggerProvider({ resource, processors: logProcessors } as any);
    logs.setGlobalLoggerProvider(loggerProvider);

    // ── 收集所有 SpanProcessors（必须在 NodeSDK 构造前准备好） ──
    // @opentelemetry/sdk-trace-base@2.x 移除了 addSpanProcessor()，
    // 所有 processor 必须在构造时通过 spanProcessors 选项一次性传入。
    const { BatchSpanProcessor, SimpleSpanProcessor } = await import("@opentelemetry/sdk-trace-base");
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const spanProcessors: any[] = [];
    // 仅当有主 OTel endpoint 时添加 BatchSpanProcessor
    if (traceExporter) {
      spanProcessors.push(new BatchSpanProcessor(traceExporter));
    }

    // ClickHouse SpanProcessor
    if (clickhouseEnabled && (globalThis as Record<string, unknown>).__chSpanExporter) {
      spanProcessors.push(
        new SimpleSpanProcessor(
          (globalThis as Record<string, unknown>).__chSpanExporter as InstanceType<typeof GrpcTraceExporter>
        )
      );
      delete (globalThis as Record<string, unknown>).__chSpanExporter;
    }

    // Langfuse 过滤型 SpanProcessor（只转发 LLM 相关 span）
    let langfuseShutdown: (() => Promise<void>) | undefined;
    try {
      const { LangfuseFilteringProcessor } =
        await import("./langfuse-span-processor.js");

      // 从 options 读取 Langfuse 配置（已经由 gateway/config.ts 从 YAML 解析好）
      let langfuseEnabled = false;
      let langfuseHost = "";
      let langfusePublicKey = "";
      let langfuseSecretKey = "";

      if (typeof options.langfuse === "object" && options.langfuse) {
        langfuseEnabled = true;
        langfuseHost = options.langfuse.host;
        langfusePublicKey = options.langfuse.publicKey;
        langfuseSecretKey = options.langfuse.secretKey;
      } else if (options.langfuse === true || process.env.LANGFUSE_ENABLED === "true") {
        // 兼容环境变量兜底
        langfuseEnabled = true;
        langfuseHost = process.env.LANGFUSE_HOST ?? "";
        langfusePublicKey = process.env.LANGFUSE_PUBLIC_KEY ?? "";
        langfuseSecretKey = process.env.LANGFUSE_SECRET_KEY ?? "";
      }

      if (langfuseEnabled && (!langfuseHost || !langfusePublicKey || !langfuseSecretKey)) {
        console.warn(
          `[core][otel] Langfuse enabled but config incomplete (host=${langfuseHost ? "✓" : "✗"}, publicKey=${langfusePublicKey ? "✓" : "✗"}, secretKey=${langfuseSecretKey ? "✓" : "✗"}). Skipping Langfuse.`,
        );
      }

      if (langfuseEnabled && langfuseHost && langfusePublicKey && langfuseSecretKey) {
        // 构造 OTLP HTTP exporter 指向 Langfuse 的 OTel 端点
        const langfuseExporter = new HttpTraceExporter({
          url: `${langfuseHost}/api/public/otel/v1/traces`,
          headers: {
            Authorization: `Basic ${Buffer.from(
              `${langfusePublicKey}:${langfuseSecretKey}`
            ).toString("base64")}`,
          },
        });

        const langfuseProcessor = new LangfuseFilteringProcessor(langfuseExporter);
        spanProcessors.push(langfuseProcessor);
        langfuseShutdown = () => langfuseProcessor.shutdown();
        console.info(
          `[core][otel] Langfuse exporter enabled ✓ | host=${langfuseHost}`,
        );
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      console.warn(`[core][otel] Langfuse exporter init failed: ${msg}. Continuing without Langfuse.`);
    }

    // 初始化 NodeSDK（不含 Metric，Metric 通过 Kafka 上报）
    // 注意：@opentelemetry/sdk-trace-base@2.x 不再支持动态 addSpanProcessor，
    // 所有 processor 必须在此处通过 spanProcessors 一次性传入。
    const sdk = new NodeSDK({
      resource,
      spanProcessors,
      contextManager: new AsyncLocalStorageContextManager(),
    });

    sdk.start();

    _sdkInstance = {
      shutdown: async () => {
        await Promise.all([
          sdk.shutdown(),
          loggerProvider.shutdown(),
          clickhouseShutdown?.(),
          langfuseShutdown?.(),
        ]);
      },
    };
    _initialized = true;

    console.info(
      `[core][otel] SDK initialized ✓ | endpoint=${endpoint || "(langfuse-only)"} | protocol=${protocol} | service=${serviceName} | tenantId=${tenantId ? tenantId.slice(0, 20) + "..." : "(none)"} | metrics=kafka`,
    );

    return true;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    diag.warn(`[core] Failed to initialize OTel SDK: ${msg}`);
    return false;
  }
}

/**
 * 优雅关闭 OTel SDK。
 */
export async function shutdownOTelSDK(): Promise<void> {
  if (!_sdkInstance) return;
  try {
    await _sdkInstance.shutdown();
  } catch {
    // Best-effort shutdown
  } finally {
    _sdkInstance = undefined;
    _initialized = false;
  }
}

/**
 * 检查 OTel SDK 是否已初始化。
 */
export function isOTelSDKInitialized(): boolean {
  return _initialized;
}

// ── Helpers ──

function parseHeadersFromEnv(): Record<string, string> {
  const raw = process.env.OTEL_EXPORTER_OTLP_HEADERS;
  if (!raw) return {};
  const headers: Record<string, string> = {};
  for (const pair of raw.split(",")) {
    const eqIdx = pair.indexOf("=");
    if (eqIdx <= 0) continue;
    headers[pair.slice(0, eqIdx).trim()] = pair.slice(eqIdx + 1).trim();
  }
  return headers;
}

function parseResourceAttributesFromEnv(): Record<string, string> {
  const raw = process.env.OTEL_RESOURCE_ATTRIBUTES;
  if (!raw) return {};
  const attrs: Record<string, string> = {};
  for (const pair of raw.split(",")) {
    const eqIdx = pair.indexOf("=");
    if (eqIdx <= 0) continue;
    attrs[pair.slice(0, eqIdx).trim()] = pair.slice(eqIdx + 1).trim();
  }
  return attrs;
}
