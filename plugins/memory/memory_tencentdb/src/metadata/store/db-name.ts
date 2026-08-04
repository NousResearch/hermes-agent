/**
 * 按实例解析元数据库名（v3.0 库级隔离）。
 * MongoDB database / SQLite 目录名均为 {mongoDbPrefix}_{sanitized_id}。
 */

/** 未配置 `mongoDbPrefix` / `TDAI_METADATA_MONGO_DB_PREFIX` 时的默认值。 */
export const DEFAULT_METADATA_DB_PREFIX = "tdai_metadata";

export class InvalidInstanceIdError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "InvalidInstanceIdError";
  }
}

/** MongoDB 库名非法字符 → `_` */
export function sanitizeInstanceIdForDb(instanceId: string): string {
  return instanceId
    .trim()
    .replace(/[/\\."$ \0]/g, "_");
}

function normalizeDbPrefix(dbPrefix?: string): string {
  const trimmed = dbPrefix?.trim();
  return trimmed || DEFAULT_METADATA_DB_PREFIX;
}

/**
 * 解析逻辑库名，如 `tdai_metadata_default`。
 * @throws InvalidInstanceIdError 空或规范化后为空
 */
export function resolveMetadataDbName(
  instanceId: string,
  dbPrefix: string = DEFAULT_METADATA_DB_PREFIX,
): string {
  const prefix = normalizeDbPrefix(dbPrefix);
  const sanitized = sanitizeInstanceIdForDb(instanceId);
  if (!sanitized) {
    throw new InvalidInstanceIdError("instance_id is empty or invalid after sanitization");
  }
  const maxInstanceLen = 64 - prefix.length - 1;
  const truncated = sanitized.slice(0, maxInstanceLen);
  return `${prefix}_${truncated}`;
}

/** SQLite：{baseDir}/{dbName}/metadata.db */
export function resolveSqliteDbPath(
  baseDir: string,
  instanceId: string,
  dbPrefix?: string,
): string {
  const dbName = resolveMetadataDbName(instanceId, dbPrefix);
  return `${baseDir.replace(/\/$/, "")}/${dbName}/metadata.db`;
}

/** SQLite 实例库目录（destroy 时递归删除） */
export function resolveSqliteDbDir(
  baseDir: string,
  instanceId: string,
  dbPrefix?: string,
): string {
  const dbName = resolveMetadataDbName(instanceId, dbPrefix);
  return `${baseDir.replace(/\/$/, "")}/${dbName}`;
}
