/**
 * InstanceConfigProvider — 实例级配置管理
 *
 * 设计要点:
 *   - VDB 配置: per-instance (每个 instanceId 独立的 VDB 连接信息), 带 TTL 缓存
 *   - COS 配置: 全局共享一份 (所有实例共用同一个 bucket, 按 pathPrefix 隔离)
 *   - 配置来源通过依赖注入的 IConfigSource 提供:
 *     - standalone: LocalConfigSource (本文件内置, 从 env vars 读取)
 *     - service:    由部署环境注入远程配置源
 *
 * 数据模型:
 *   Core 进程
 *     ├── COS: 全局一份 { cosUrl, tmpSecretId, tmpSecretKey, tmpToken, expirationTime, pathPrefix }
 *     └── VDB 池 (Map<instanceId, VdbConfig>):
 *         ├── inst-001 → { url: vdb-1, apiKey: xxx, database: db1 }
 *         ├── inst-002 → { url: vdb-2, apiKey: yyy, database: db2 }
 *         └── inst-003 → { url: vdb-1, apiKey: xxx, database: db3 }
 */

import type { IConfigSource } from "./abstractions/index.js";
import { readVdbEnvConfig, readCosEnvConfig } from "../utils/env-config.js";

// ════════════════════════════════════════════════════════
// Types
// ════════════════════════════════════════════════════════

export interface VdbConfig {
  url: string;
  user: string;
  apiKey: string;
  database: string;
}

export interface CosConfig {
  cosUrl: string;
  tmpSecretId: string;
  tmpSecretKey: string;
  tmpToken: string;
  /** ISO 8601 过期时间 (仅临时凭证模式有效) */
  expirationTime: string;
  pathPrefix: string;
}

export interface InstanceConfig {
  instanceId: string;
  vdb: VdbConfig;
  cos: CosConfig | null;
}

interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

// Re-export interface for backward compatibility (consumers may still
// import IConfigSource from this module — the canonical location is now
// src/core/abstractions/).
export type { IConfigSource };

// ════════════════════════════════════════════════════════
// LocalConfigSource — 默认实现 (open-source / standalone)
// ════════════════════════════════════════════════════════
//
// 从进程环境变量读取 VDB + COS 配置。适合无管控面的单租户自部署场景。
// 与接口同居一处，遵循项目现有约定 (cf. MockCredentialProvider 与
// ICredentialProvider 同写在 src/core/storage/credential-provider.ts)。
//
// Environment variables:
//   VDB_ENDPOINT, VDB_USER, VDB_API_KEY, VDB_DATABASE
//   COS_SECRET_ID, COS_SECRET_KEY, COS_TOKEN, COS_URL, COS_PATH_PREFIX

export class LocalConfigSource implements IConfigSource {
  // Logger kept for future diagnostic logging; constructor signature mirrors
  // remote sources so callers can swap implementations interchangeably.
  constructor(private readonly _logger: Logger) {
    void this._logger;
  }

  async fetchVdb(_instanceId: string): Promise<VdbConfig> {
    return readVdbEnvConfig();
  }

  async fetchCos(): Promise<CosConfig | null> {
    const cfg = readCosEnvConfig();
    if (!cfg) return null;
    return {
      cosUrl: cfg.cosUrl,
      tmpSecretId: cfg.tmpSecretId,
      tmpSecretKey: cfg.tmpSecretKey,
      tmpToken: cfg.tmpToken,
      expirationTime: "",
      pathPrefix: cfg.pathPrefix,
    };
  }
}

// ════════════════════════════════════════════════════════
// VDB 缓存条目
// ════════════════════════════════════════════════════════

interface VdbCacheEntry {
  config: VdbConfig;
  expiresAt: number;
  lastAccessedAt: number;
}

// ════════════════════════════════════════════════════════
// InstanceConfigProvider
// ════════════════════════════════════════════════════════

export interface InstanceConfigProviderOptions {
  /**
   * Pre-constructed config source for the current deployment.
   */
  source: IConfigSource;
  /** VDB 缓存 TTL (毫秒), 默认 5 分钟 */
  vdbTtlMs?: number;
  /** COS 凭证提前刷新时间 (毫秒), 默认 2 分钟 */
  cosBufferMs?: number;
  /** 最大缓存实例数, 超出后 LRU 淘汰, 默认 1000 */
  maxInstances?: number;
  logger: Logger;
}

export class InstanceConfigProvider {
  private source: IConfigSource;
  private logger: Logger;

  // ── VDB: per-instance 缓存 ──
  private vdbPool = new Map<string, VdbCacheEntry>();
  private vdbTtlMs: number;
  private maxInstances: number;
  /**
   * Per-instance in-flight fetch dedupe (H-2 fix):
   * 并发首次访问同一 instanceId 时，复用同一个 fetch Promise，
   * 避免向 source 同时发出 N 次请求触发限流。
   */
  private vdbFetchPromises = new Map<string, Promise<VdbConfig>>();

  // ── COS: 全局单例缓存 (一份凭证，按 PathPrefix 隔离) ──
  private cosCache: CosConfig | null = null;
  private cosExpiresAt = 0;
  private cosBufferMs: number;
  private cosFetchPromise: Promise<CosConfig | null> | null = null;

  constructor(opts: InstanceConfigProviderOptions) {
    this.logger = opts.logger;
    this.vdbTtlMs = opts.vdbTtlMs ?? 5 * 60 * 1000;
    this.cosBufferMs = opts.cosBufferMs ?? 2 * 60 * 1000;
    this.maxInstances = opts.maxInstances ?? 1000;
    this.source = opts.source;
    this.logger.info(`[instance-config] InstanceConfigProvider initialized (source=${opts.source.constructor.name})`);
  }

  // ════════════════════════════════════════════════════════
  // Public API
  // ════════════════════════════════════════════════════════

  /**
   * 获取指定实例的完整配置 (VDB per-instance + COS 全局)
   */
  async resolve(instanceId: string): Promise<InstanceConfig> {
    const [vdb, cos] = await Promise.all([
      this.resolveVdb(instanceId),
      this.resolveCos(),
    ]);
    return { instanceId, vdb, cos };
  }
  /**
   * 获取指定实例的 VDB 配置 (带缓存)
   *
   * 策略：
   * 1. 缓存命中且未过期 → 直接返回（并刷新 LRU 位置）
   * 2. 缓存为空或已过期 → 从 source 获取（并发请求同一 instanceId 时 in-flight 去重）
   * 3. source 返回空/错误 → 直接报错并记录日志（不缓存空值）
   */
  async resolveVdb(instanceId: string): Promise<VdbConfig> {
    const now = Date.now();
    const cached = this.vdbPool.get(instanceId);

    if (cached && now < cached.expiresAt) {
      cached.lastAccessedAt = now;
      // LRU 重排 (H-3): 把 entry 移到 Map 末尾, 使 evict 时取首元素即为 LRU。
      // delete+set 是 V8 上 Map 的 O(1) 操作。
      this.vdbPool.delete(instanceId);
      this.vdbPool.set(instanceId, cached);
      return cached.config;
    }

    // 缓存未命中或已过期 → 进入 fetch 路径，先去重再请求 (H-2)
    const inflight = this.vdbFetchPromises.get(instanceId);
    if (inflight) {
      this.logger.debug?.(`[instance-config] VDB fetch in-flight for ${instanceId}, awaiting...`);
      return inflight;
    }

    this.logger.debug?.(`[instance-config] VDB cache ${cached ? "expired" : "miss"} for ${instanceId}, fetching...`);
    const fetchPromise = this.fetchAndStoreVdb(instanceId);
    this.vdbFetchPromises.set(instanceId, fetchPromise);
    try {
      return await fetchPromise;
    } finally {
      // 清理 in-flight 标记。注意要在 await 之后清理 (即使 fetch 抛错也清), 
      // 否则一次失败会让该 instanceId 永久卡住。
      this.vdbFetchPromises.delete(instanceId);
    }
  }

  /**
   * 实际执行 source fetch + 写入 vdbPool。
   * 仅由 resolveVdb 内部调用 (并发去重保证只跑一次)。
   */
  private async fetchAndStoreVdb(instanceId: string): Promise<VdbConfig> {
    const config = await this.source.fetchVdb(instanceId);

    // source 返回空 → 直接报错记录日志
    if (!config || !config.url) {
      const msg = `[instance-config] Config source returned empty VDB config for instanceId="${instanceId}" (url=${config?.url})`;
      this.logger.error(msg);
      throw new Error(msg);
    }

    // LRU 淘汰
    if (this.vdbPool.size >= this.maxInstances && !this.vdbPool.has(instanceId)) {
      this.evictLru();
    }

    const now = Date.now();
    this.vdbPool.set(instanceId, {
      config,
      expiresAt: now + this.vdbTtlMs,
      lastAccessedAt: now,
    });

    return config;
  }

  /**
   * 获取全局 COS 配置 (带缓存, 自动续期临时凭证)
   */
  async resolveCos(): Promise<CosConfig | null> {
    const now = Date.now();

    // 缓存有效 → 直接返回
    if (this.cosCache && now < this.cosExpiresAt) {
      return this.cosCache;
    }

    // 防止并发请求同时刷新
    if (this.cosFetchPromise) {
      return this.cosFetchPromise;
    }

    this.cosFetchPromise = this.refreshCos(now).finally(() => {
      this.cosFetchPromise = null;
    });

    return this.cosFetchPromise;
  }

  /**
   * 清除指定实例的 VDB 缓存 (实例下线时调用)
   */
  evictVdb(instanceId: string): void {
    this.vdbPool.delete(instanceId);
    this.logger.debug?.(`[instance-config] Evicted VDB cache for ${instanceId}`);
  }

  /**
   * 强制刷新 COS 凭证
   */
  async refreshCosNow(): Promise<CosConfig | null> {
    return this.refreshCos(Date.now());
  }

  /**
   * 清除所有缓存
   */
  clear(): void {
    this.vdbPool.clear();
    this.cosCache = null;
    this.cosExpiresAt = 0;
    this.logger.info(`[instance-config] All caches cleared`);
  }

  /**
   * 当前缓存的实例数
   */
  get poolSize(): number {
    return this.vdbPool.size;
  }

  /**
   * 当前 COS 凭证是否有效
   */
  get isCosValid(): boolean {
    return this.cosCache !== null && Date.now() < this.cosExpiresAt;
  }

  // ════════════════════════════════════════════════════════
  // Internal
  // ════════════════════════════════════════════════════════

  private async refreshCos(now: number): Promise<CosConfig | null> {
    this.logger.debug?.(`[instance-config] Refreshing COS config...`);
    try {
      const cos = await this.source.fetchCos();
      this.cosCache = cos;
      this.cosExpiresAt = this.calcCosExpiry(cos, now);
      return cos;
    } catch (e) {
      this.logger.warn(`[instance-config] Failed to refresh COS config: ${e}`);
      // 如果旧缓存还在，延长一小段时间继续使用（降级）
      if (this.cosCache) {
        this.cosExpiresAt = now + 30_000; // 30s 后重试
        this.logger.warn(`[instance-config] Using stale COS config for 30s`);
        return this.cosCache;
      }
      return null;
    }
  }

  /**
   * 计算 COS 缓存过期时间:
   *   - 有 expirationTime: min(服务端过期时间 - buffer, vdbTtl)
   *   - 无 expirationTime: 使用 vdbTtl (本地长期凭证场景)
   */
  private calcCosExpiry(cos: CosConfig | null, now: number): number {
    if (!cos?.expirationTime) {
      return now + this.vdbTtlMs;
    }
    const serverExpiry = new Date(cos.expirationTime).getTime();
    if (isNaN(serverExpiry)) {
      return now + this.vdbTtlMs;
    }
    return Math.min(serverExpiry - this.cosBufferMs, now + this.vdbTtlMs);
  }

  /**
   * LRU 淘汰: 删除最近最少访问的实例。
   *
   * 实现说明 (H-3): 利用 Map 的插入顺序就是访问顺序的特性 —— resolveVdb 在 cache hit
   * 时已经做了 delete+set 把热 key 移到 Map 末尾, 所以 Map 的首元素就是 LRU,
   * 直接取第一个 key 即可, O(1)。
   */
  private evictLru(): void {
    const firstKey = this.vdbPool.keys().next().value;
    if (firstKey !== undefined) {
      this.vdbPool.delete(firstKey);
      this.logger.debug?.(`[instance-config] LRU evicted VDB cache for ${firstKey}`);
    }
  }
}
