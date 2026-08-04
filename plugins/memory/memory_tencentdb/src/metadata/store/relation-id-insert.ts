/**
 * 关联表 `id` 列插入：碰撞检测与重试（配合 generateRelationId）。
 */
import { generateRelationId } from "../utils/id-generator.js";

export const RELATION_ID_RETRY_LIMIT = 3;

/** SQLite：关联表主键 `id` 唯一约束冲突。 */
export function isSqliteRelationIdCollision(err: unknown): boolean {
  const msg = err instanceof Error ? err.message : String(err);
  return /UNIQUE constraint failed: meta_\w+\.id\b/.test(msg);
}

/** MongoDB：关联表主键 `id` 重复键（E11000）。 */
export function isMongoRelationIdCollision(err: unknown): boolean {
  if (typeof err !== "object" || err === null) return false;
  const e = err as { code?: number; keyPattern?: Record<string, unknown> };
  return e.code === 11000 && Boolean(e.keyPattern?.id);
}

/**
 * 使用自动生成的 relation id 执行插入；`fixedId` 已指定时不重试。
 */
export function runWithGeneratedRelationId<T>(
  fixedId: string | undefined,
  isCollision: (err: unknown) => boolean,
  insert: (id: string) => T,
): T {
  if (fixedId) return insert(fixedId);
  let lastErr: unknown;
  for (let attempt = 0; attempt < RELATION_ID_RETRY_LIMIT; attempt++) {
    try {
      return insert(generateRelationId());
    } catch (err) {
      if (isCollision(err)) {
        lastErr = err;
        continue;
      }
      throw err;
    }
  }
  if (lastErr instanceof Error) throw lastErr;
  throw new Error("relation id collision after max retries");
}
