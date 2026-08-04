/**
 * 元数据存储工厂 + 配置解析 + MetadataStorePool（v3.0 按实例分库）。
 */

import { rm } from "node:fs/promises";
import type { IMetadataStore, MetadataBackend } from "./interface.js";
import { SqliteMetadataStore } from "./sqlite-adapter.js";
import {
  DEFAULT_METADATA_DB_PREFIX,
  resolveMetadataDbName,
  resolveSqliteDbDir,
  resolveSqliteDbPath,
} from "./db-name.js";

export interface MetadataStoreConfig {
  backend: MetadataBackend;
  /** SQLite 根目录（backend=sqlite）。每个实例：{baseDir}/tdai_metadata_{id}/metadata.db */
  sqliteBaseDir?: string;
  /** MongoDB 连接串（backend=mongodb）。 */
  mongoUri?: string;
  /** MongoDB 是否启用事务（默认 true，需副本集）。 */
  mongoTransactions?: boolean;
  /** 内存中最多缓存多少个实例 store 连接（LRU 驱逐，仅 close 不删库）。 */
  storeCacheMaxInstances?: number;
  /** 元数据库名前缀，默认 `tdai_metadata`；完整库名 `{prefix}_{instance_id}`。 */
  mongoDbPrefix?: string;
}

export interface PurgeMetadataResult {
  db_name: string;
  dropped: boolean;
}

export type MetadataDeployMode = "standalone" | "service";

const DEFAULT_SQLITE_BASE = "./data/metadata";
const DEFAULT_STORE_CACHE_MAX = 128;

function hasExplicitMongoUri(env: NodeJS.ProcessEnv): boolean {
  return !!env.TDAI_METADATA_MONGO_URI?.trim();
}

function hasExplicitSqliteBaseDir(env: NodeJS.ProcessEnv): boolean {
  return !!env.TDAI_METADATA_SQLITE_BASE_DIR?.trim();
}

/**
 * Mongo 与 SQLite 根目录不可同时显式配置（env / yaml 回填后校验）。
 */
export function assertMetadataStoreConfigExclusive(env: NodeJS.ProcessEnv = process.env): void {
  if (hasExplicitMongoUri(env) && hasExplicitSqliteBaseDir(env)) {
    throw new MetadataStartupValidationError(
      "Metadata startup validation failed: set either TDAI_METADATA_MONGO_URI or " +
        "TDAI_METADATA_SQLITE_BASE_DIR, not both",
    );
  }
}

/**
 * 从环境变量解析存储配置。
 *
 * 推断规则（v3.0）：
 *   - TDAI_METADATA_MONGO_URI 非空 → mongodb
 *   - 否则 → sqlite（显式 TDAI_METADATA_SQLITE_BASE_DIR 或 fallback）
 *   - 二者同时显式配置 → 启动报错（见 assertMetadataStoreConfigExclusive）
 *
 * deployMode=service 时须 mongodb（见 validateMetadataStartupConfig）。
 *
 * 废弃：TDAI_METADATA_BACKEND、TDAI_METADATA_MONGO_DB、TDAI_METADATA_SQLITE_PATH
 */
export function loadStoreConfig(
  env: NodeJS.ProcessEnv = process.env,
  fallbackSqliteBaseDir?: string,
): MetadataStoreConfig {
  assertMetadataStoreConfigExclusive(env);

  const mongoUri = env.TDAI_METADATA_MONGO_URI?.trim();
  const cacheMax = parseInt(env.TDAI_METADATA_STORE_CACHE_MAX ?? "", 10);
  const storeCacheMaxInstances =
    Number.isFinite(cacheMax) && cacheMax > 0 ? cacheMax : DEFAULT_STORE_CACHE_MAX;

  const mongoDbPrefix =
    env.TDAI_METADATA_MONGO_DB_PREFIX?.trim() || DEFAULT_METADATA_DB_PREFIX;

  if (mongoUri) {
    return {
      backend: "mongodb",
      mongoUri,
      mongoTransactions: env.TDAI_METADATA_MONGO_TRANSACTIONS !== "false",
      storeCacheMaxInstances,
      mongoDbPrefix,
    };
  }

  return {
    backend: "sqlite",
    sqliteBaseDir: env.TDAI_METADATA_SQLITE_BASE_DIR ?? fallbackSqliteBaseDir ?? DEFAULT_SQLITE_BASE,
    storeCacheMaxInstances,
    mongoDbPrefix,
  };
}

/** service 模式元数据配置校验失败时抛出，Gateway 启动应 fail-fast。 */
export class MetadataStartupValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "MetadataStartupValidationError";
  }
}

export function validateMetadataStartupConfig(
  deployMode: MetadataDeployMode,
  env: NodeJS.ProcessEnv = process.env,
  fallbackSqliteBaseDir?: string,
): MetadataStoreConfig {
  const config = loadStoreConfig(env, fallbackSqliteBaseDir);
  if (deployMode !== "service") {
    return config;
  }

  const errors: string[] = [];
  if (!config.mongoUri?.trim()) {
    errors.push("TDAI_METADATA_MONGO_URI is required when deployMode=service");
  }
  if (errors.length > 0) {
    throw new MetadataStartupValidationError(
      `Metadata startup validation failed: ${errors.join("; ")}`,
    );
  }
  return config;
}

/**
 * 构造并初始化单个实例库的 IMetadataStore。
 */
export async function createMetadataStore(
  config: MetadataStoreConfig,
  instanceId: string,
): Promise<IMetadataStore> {
  switch (config.backend) {
    case "sqlite": {
      const baseDir = config.sqliteBaseDir ?? DEFAULT_SQLITE_BASE;
      const dbPath = resolveSqliteDbPath(baseDir, instanceId, config.mongoDbPrefix);
      const store = new SqliteMetadataStore(dbPath);
      await store.init();
      return store;
    }
    case "mongodb": {
      if (!config.mongoUri) {
        throw new Error("TDAI_METADATA_MONGO_URI is required when backend=mongodb");
      }
      const { MongoClient } = await import("mongodb");
      const { MongoMetadataStore } = await import("./mongodb-adapter.js");
      const client = new MongoClient(config.mongoUri);
      await client.connect();
      const dbName = resolveMetadataDbName(instanceId, config.mongoDbPrefix);
      const store = new MongoMetadataStore(client, dbName, {
        useTransactions: config.mongoTransactions ?? true,
        ownsClient: true,
      });
      await store.init();
      return store;
    }
    case "mysql":
      throw new Error("MySQL backend not yet implemented");
    default:
      throw new Error(`Unknown metadata backend: ${config.backend as string}`);
  }
}

interface CachedStore {
  instanceId: string;
  store: IMetadataStore;
  /** mongodb 时持有 client 引用以便 close */
  mongoClient?: import("mongodb").MongoClient;
}

/**
 * 按实例懒建库、LRU 缓存、purge dropDatabase。
 */
export class MetadataStorePool {
  private readonly cache = new Map<string, CachedStore>();
  private readonly config: MetadataStoreConfig;
  private sharedMongoClient: import("mongodb").MongoClient | null = null;
  private sharedMongoClientPromise: Promise<import("mongodb").MongoClient> | null = null;

  constructor(config: MetadataStoreConfig) {
    this.config = config;
  }

  get backend(): MetadataBackend {
    return this.config.backend;
  }

  private get dbPrefix(): string {
    return this.config.mongoDbPrefix?.trim() || DEFAULT_METADATA_DB_PREFIX;
  }

  private async getSharedMongoClient(): Promise<import("mongodb").MongoClient> {
    if (this.sharedMongoClient) return this.sharedMongoClient;
    if (!this.sharedMongoClientPromise) {
      this.sharedMongoClientPromise = (async () => {
        if (!this.config.mongoUri) throw new Error("TDAI_METADATA_MONGO_URI is required");
        const { MongoClient } = await import("mongodb");
        const client = new MongoClient(this.config.mongoUri);
        await client.connect();
        this.sharedMongoClient = client;
        return client;
      })();
    }
    return this.sharedMongoClientPromise;
  }

  private touchLru(instanceId: string, entry: CachedStore): void {
    this.cache.delete(instanceId);
    this.cache.set(instanceId, entry);
    this.evictIfNeeded();
  }

  private evictIfNeeded(): void {
    const max = this.config.storeCacheMaxInstances ?? DEFAULT_STORE_CACHE_MAX;
    while (this.cache.size > max) {
      const oldestKey = this.cache.keys().next().value as string | undefined;
      if (!oldestKey) break;
      const oldest = this.cache.get(oldestKey);
      this.cache.delete(oldestKey);
      if (oldest) {
        void Promise.resolve(oldest.store.close()).catch(() => {});
      }
    }
  }

  async getStore(instanceId: string): Promise<IMetadataStore> {
    const existing = this.cache.get(instanceId);
    if (existing) {
      this.touchLru(instanceId, existing);
      return existing.store;
    }

    if (this.config.backend === "mongodb") {
      const client = await this.getSharedMongoClient();
      const { MongoMetadataStore } = await import("./mongodb-adapter.js");
      const dbName = resolveMetadataDbName(instanceId, this.dbPrefix);
      const store = new MongoMetadataStore(client, dbName, {
        useTransactions: this.config.mongoTransactions ?? true,
        ownsClient: false,
      });
      await store.init();
      const entry: CachedStore = { instanceId, store, mongoClient: client };
      this.cache.set(instanceId, entry);
      this.evictIfNeeded();
      return store;
    }

    const store = await createMetadataStore(this.config, instanceId);
    const entry: CachedStore = { instanceId, store };
    this.cache.set(instanceId, entry);
    this.evictIfNeeded();
    return store;
  }

  async purgeInstance(instanceId: string): Promise<PurgeMetadataResult> {
    const dbName = resolveMetadataDbName(instanceId, this.dbPrefix);
    const cached = this.cache.get(instanceId);
    if (cached) {
      await Promise.resolve(cached.store.close()).catch(() => {});
      this.cache.delete(instanceId);
    }

    if (this.config.backend === "mongodb") {
      const client = await this.getSharedMongoClient();
      await client.db(dbName).dropDatabase();
      return { db_name: dbName, dropped: true };
    }

    const baseDir = this.config.sqliteBaseDir ?? DEFAULT_SQLITE_BASE;
    const dir = resolveSqliteDbDir(baseDir, instanceId, this.dbPrefix);
    await rm(dir, { recursive: true, force: true });
    return { db_name: dbName, dropped: true };
  }

  async closeAll(): Promise<void> {
    for (const [key, entry] of this.cache) {
      await Promise.resolve(entry.store.close()).catch(() => {});
      this.cache.delete(key);
    }
    if (this.sharedMongoClient) {
      await this.sharedMongoClient.close().catch(() => {});
      this.sharedMongoClient = null;
      this.sharedMongoClientPromise = null;
    }
  }
}

export async function createMetadataStorePool(
  config: MetadataStoreConfig,
): Promise<MetadataStorePool> {
  return new MetadataStorePool(config);
}
