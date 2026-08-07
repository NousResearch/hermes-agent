/**
 * State Backend — 接口 + 默认实现导出 + 后端工厂。
 *
 * 默认实现 (LocalStateBackend) 跟接口同住 core，自带可用。
 * 远程状态后端在运行时按需动态加载；如果当前构建未包含对应实现，
 * 当配置要求使用远程后端时会抛出明确错误。
 */

export type {
  IStateBackend,
  PipelineSessionState,
  TimerEntry,
  TaskPayload,
  CaptureAtomicParams,
  CaptureAtomicResult,
} from "./types.js";
export { DEFAULT_PIPELINE_STATE } from "./types.js";

export { LocalStateBackend } from "./local-backend.js";

import type { IStateBackend, TimerEntry } from "./types.js";
import { LocalStateBackend } from "./local-backend.js";

export interface StateBackendConfig {
  type: "local" | "redis";
  local?: {
    onTimerExpired?: (entry: TimerEntry) => void;
  };
  redis?: {
    /** backend connection URL */
    url?: string;
    host?: string;
    port?: number;
    password?: string;
    /** database index (default: 0) */
    db?: number;
    keyPrefix?: string;
    consumerGroup?: string;
  };
}

/**
 * 工厂函数：根据配置创建对应的 State Backend。
 *
 * - type === "local": 内置 LocalStateBackend，零外部依赖
 * - remote backend: 动态加载远程状态后端实现；如果当前构建未包含，
 *   抛出明确错误。
 */
export async function createStateBackend(config: StateBackendConfig): Promise<IStateBackend> {
  if (config.type === "redis") {
    const redisCfg = config.redis;
    if (!redisCfg) throw new Error("redis config is required when state_backend=redis");

    let RedisStateBackendCtor: typeof import("../../integrations/redis/index.js").RedisStateBackend;
    try {
      ({ RedisStateBackend: RedisStateBackendCtor } = await import("../../integrations/redis/index.js"));
    } catch (err) {
      throw new Error(
        "[state-backend] Redis integration is not available — install or initialize " +
        "src/integrations/redis/ (private submodule) to use state_backend=redis, " +
        "or switch to state_backend=local. " +
        `Original error: ${err instanceof Error ? err.message : String(err)}`,
      );
    }

    // Dynamically import the remote backend client only when needed.
    const { default: Redis } = await import("ioredis");

    let client;
    if (redisCfg.url) {
      client = new Redis(redisCfg.url);
    } else {
      client = new Redis({
        host: redisCfg.host ?? "127.0.0.1",
        port: redisCfg.port ?? 6379,
        password: redisCfg.password,
        db: redisCfg.db ?? 0,
      });
    }

    const backend = new RedisStateBackendCtor({
      client: client as never,
      keyPrefix: redisCfg.keyPrefix,
      consumerGroup: redisCfg.consumerGroup,
    });
    await backend.initialize();
    return backend;
  }

  // default: local
  const backend = new LocalStateBackend(config.local);
  await backend.initialize?.();
  return backend;
}
