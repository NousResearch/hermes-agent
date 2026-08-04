/**
 * Observability Backend Factory — 工厂函数 + 全局单例管理。
 *
 * 通过配置驱动创建可观测性后端实例：
 * - "noop"     → NoopObservabilityBackend（默认，零开销）
 * - "console"  → ConsoleObservabilityBackend（开发调试，输出到 stdout）
 * - "otlp"     → OtlpObservabilityBackend（开源用户推荐，标准 OTLP 协议）
 * - "internal" → 动态加载私有模块（内部环境：智研 + Kafka + Langfuse）
 *
 * 开源用户推荐使用 "otlp" 类型，只需配置一个 endpoint 即可将
 * Trace/Log/Metric 全部上报到任何支持 OTLP 协议的后端。
 *
 * 参考 src/core/storage/factory.ts 的设计模式。
 */

import type { IObservabilityBackend, ObservabilityConfig } from "./types.js";
import { NoopObservabilityBackend } from "./noop-backend.js";
import { ConsoleObservabilityBackend } from "./console-backend.js";
import { OtlpObservabilityBackend } from "./otlp-backend.js";

const TAG = "[observability][factory]";

// ============================
// 动态加载私有模块
// ============================

/**
 * 尝试动态加载可选可观测性模块。
 * 加载失败返回 null。
 */
async function loadInternalBackend(): Promise<{ createInternalObservabilityBackend: (config: ObservabilityConfig) => Promise<IObservabilityBackend> } | null> {
  try {
    return await import("../../integrations/observability/index.js");
  } catch {
    return null;
  }
}

// ============================
// 工厂函数
// ============================

/**
 * 创建可观测性后端实例。
 *
 * @param config 可观测性配置
 * @returns IObservabilityBackend 实例
 */
export async function createObservabilityBackend(
  config: ObservabilityConfig,
): Promise<IObservabilityBackend> {
  const type = config.type ?? "noop";

  switch (type) {
    case "internal": {
      const privateModule = await loadInternalBackend();
      if (!privateModule) {
        console.warn(
          `${TAG} Internal observability backend requested but private module not available. ` +
          `Falling back to console backend. Install the private submodule for full observability.`,
        );
        const backend = new ConsoleObservabilityBackend();
        await backend.initialize(config);
        return backend;
      }
      const backend = await privateModule.createInternalObservabilityBackend(config);
      await backend.initialize(config);
      return backend;
    }

    case "otlp": {
      const backend = new OtlpObservabilityBackend();
      await backend.initialize(config);
      return backend;
    }

    case "console": {
      const backend = new ConsoleObservabilityBackend();
      await backend.initialize(config);
      return backend;
    }

    case "noop":
    default: {
      const backend = new NoopObservabilityBackend();
      await backend.initialize(config);
      return backend;
    }
  }
}

// ============================
// 全局单例管理
// ============================

/** 全局可观测性后端实例 */
let _globalBackend: IObservabilityBackend = new NoopObservabilityBackend();

/** 是否已初始化 */
let _initialized = false;

/**
 * 获取全局可观测性后端实例。
 * 如果尚未初始化，返回 Noop 实例（安全降级）。
 */
export function getObservabilityBackend(): IObservabilityBackend {
  return _globalBackend;
}

/**
 * 初始化全局可观测性后端。
 * 幂等：多次调用只有第一次生效。
 *
 * @param config 可观测性配置
 */
export async function initObservabilityBackend(config: ObservabilityConfig): Promise<void> {
  if (_initialized) return;

  try {
    _globalBackend = await createObservabilityBackend(config);
    _initialized = true;
    console.log(`${TAG} Observability backend initialized: type=${_globalBackend.type}`);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    console.warn(`${TAG} Failed to initialize observability backend: ${msg}. Using noop.`);
    _globalBackend = new NoopObservabilityBackend();
    _initialized = true;
  }
}

/**
 * 重置全局可观测性后端（用于插件热重载和测试）。
 * 会先调用当前后端的 shutdown()。
 */
export async function resetObservabilityBackend(): Promise<void> {
  try {
    await _globalBackend.shutdown();
  } catch {
    // 静默
  }
  _globalBackend = new NoopObservabilityBackend();
  _initialized = false;
}
