/**
 * Observability Module — Barrel Export
 *
 * 统一导出可观测性模块的所有公开类型、接口、工厂函数和默认实现。
 * 调用方通过此文件导入所需的可观测性能力。
 */

// ============================
// Types & Interfaces
// ============================
export type {
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
  LLMTraceConfig,
  OTelConfig,
  ClickHouseConfig,
  ObservabilityConfig,
  ObservabilityLogger,
} from "./types.js";

// ============================
// Factory & Singleton
// ============================
export {
  createObservabilityBackend,
  getObservabilityBackend,
  initObservabilityBackend,
  resetObservabilityBackend,
} from "./factory.js";

// ============================
// Default Implementations (Open-source)
// ============================
export {
  NoopObservabilityBackend,
  NoopTraceBackend,
  NoopLogBackend,
  NoopMetricBackend,
  NoopLLMTraceBackend,
  NoopTraceMiddleware,
  NoopTracePropagation,
} from "./noop-backend.js";

export {
  ConsoleObservabilityBackend,
  ConsoleTraceBackend,
  ConsoleLogBackend,
  ConsoleMetricBackend,
  ConsoleLLMTraceBackend,
  ConsoleTraceMiddleware,
  ConsoleTracePropagation,
} from "./console-backend.js";

export {
  OtlpObservabilityBackend,
  OtlpTraceBackend,
  OtlpLogBackend,
  OtlpMetricBackend,
  OtlpLLMTraceBackend,
  OtlpTraceMiddleware,
  OtlpTracePropagation,
} from "./otlp-backend.js";

// ============================
// Facade Modules (backward-compatible)
// ============================
export { trace } from "./trace.js";
export { log } from "./log.js";
export { obsLogger } from "./obs-logger.js";
export { metricProducer, calculatePartition } from "./kafka-metric-producer.js";
export { wrapWithTrace, startChildSpan, withSpan } from "./trace-middleware.js";
export { serializeTraceContext, deserializeTraceContext } from "./trace-propagation.js";
export { TracedTaskExecutor } from "./traced-task-executor.js";
export {
  LangfuseFilteringProcessor,
  parseLangfuseConfig,
  isLLMRelatedSpan,
  createLangfuseSpanProcessor,
} from "./langfuse-span-processor.js";
export type { LangfuseConfig, LangfuseConfigEnabled, LangfuseConfigDisabled } from "./langfuse-span-processor.js";

// ============================
// Metric Tracking (unchanged)
// ============================
export {
  MetricTrackingRunner,
  MetricTrackingRunnerFactory,
  taskIdToMetricName,
  computeCredit,
  estimateCreditFromChars,
  TOKENS_PER_CREDIT,
  INPUT_RATE,
  OUTPUT_RATE,
  CACHE_RATE,
  DEFAULT_MULTIPLIER,
} from "./metric-tracking-runner.js";
export type { LLMUsage, LLMRunnerWithUsage } from "./metric-tracking-runner.js";
