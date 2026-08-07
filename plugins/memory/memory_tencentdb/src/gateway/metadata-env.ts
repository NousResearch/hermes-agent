/**
 * 将 Gateway yaml `metadata` 块回填到 process.env（仅当 env 未设置时）。
 *
 * 解析优先级：process.env > yaml（与 TDAI_METADATA_MAX_USERS 一致）。
 * factory.ts 仍只读 env，无需改调用签名。
 */

import type { GatewayMetadataConfig } from "./config.js";

function setEnvIfEmpty(key: string, value: string | undefined): void {
  if (value?.trim() && !process.env[key]?.trim()) {
    process.env[key] = value.trim();
  }
}

/**
 * 在 validateMetadataStartupConfig / ensureMetadataStore 之前调用。
 */
export function applyMetadataEnvFromGatewayConfig(metadata: GatewayMetadataConfig): void {
  const store = metadata.store;
  if (store) {
    setEnvIfEmpty("TDAI_METADATA_SQLITE_BASE_DIR", store.sqliteBaseDir);
    setEnvIfEmpty("TDAI_METADATA_MONGO_URI", store.mongoUri);
    setEnvIfEmpty("TDAI_METADATA_MONGO_DB_PREFIX", store.mongoDbPrefix);
    if (store.mongoTransactions === false && !process.env.TDAI_METADATA_MONGO_TRANSACTIONS?.trim()) {
      process.env.TDAI_METADATA_MONGO_TRANSACTIONS = "false";
    }
    if (
      store.storeCacheMaxInstances != null &&
      !process.env.TDAI_METADATA_STORE_CACHE_MAX?.trim()
    ) {
      process.env.TDAI_METADATA_STORE_CACHE_MAX = String(store.storeCacheMaxInstances);
    }
  }

  const memory = metadata.systemUser?.memory;
  if (memory) {
    setEnvIfEmpty("TDAI_MEMORY_SYSTEM_USER_ID", memory.userId);
    setEnvIfEmpty("TDAI_MEMORY_SYSTEM_USER_NAME", memory.displayName);
    setEnvIfEmpty("TDAI_MEMORY_SYSTEM_USER_KEY", memory.userKey);
  }
}
