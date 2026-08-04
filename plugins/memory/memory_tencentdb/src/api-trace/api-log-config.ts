/**
 * API trace 运行时配置：档位由元数据 store backend 推断；默认写 stdout JSON。
 */
import type { MetadataBackend } from "../metadata/store/interface.js";
import { resolvePolicy, type ApiTracePolicy } from "./api-trace-policy.js";

export interface ApiTraceLogConfig {
  enabled: boolean;
}

export interface ApiTraceInitOptions {
  enabled?: boolean;
}

export interface ApiTraceRuntimeConfig {
  log: ApiTraceLogConfig;
  policy: ApiTracePolicy;
}

let runtimeConfig: ApiTraceRuntimeConfig | null = null;

function buildConfig(
  backend: MetadataBackend = "sqlite",
  opts?: ApiTraceInitOptions,
): ApiTraceRuntimeConfig {
  return {
    log: { enabled: opts?.enabled ?? true },
    policy: resolvePolicy(backend),
  };
}

/** Gateway 启动时注入元数据存储 backend（决定 full/lite）。 */
export function initApiTraceConfig(
  metadataBackend: MetadataBackend = "sqlite",
  opts?: ApiTraceInitOptions,
): void {
  runtimeConfig = buildConfig(metadataBackend, opts);
}

export function getApiTraceConfig(): ApiTraceRuntimeConfig {
  if (!runtimeConfig) {
    runtimeConfig = buildConfig("sqlite");
  }
  return runtimeConfig;
}

/** 测试用：重置缓存配置。 */
export function resetApiTraceConfigForTests(): void {
  runtimeConfig = null;
}

export function isApiTraceActive(): boolean {
  return getApiTraceConfig().log.enabled;
}

export { resolvePolicy, resolveProfile, type ApiTracePolicy } from "./api-trace-policy.js";
