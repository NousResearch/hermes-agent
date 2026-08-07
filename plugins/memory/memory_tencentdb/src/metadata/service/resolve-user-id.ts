/**
 * 将 user_id 或 user_key 解析为 user_id（权限/查询类接口共用）。
 */
import type { MetadataService } from "./metadata-service.js";
import { MetadataError } from "./metadata-service.js";

export async function resolveUserId(
  svc: MetadataService,
  ids: { user_id?: string; user_key?: string },
): Promise<string> {
  if (ids.user_id) return ids.user_id;
  if (ids.user_key) {
    const u = await svc.getUserByKey(ids.user_key);
    if (!u) throw new MetadataError("user_not_found", "user not found for user_key");
    return u.user_id;
  }
  throw new MetadataError("invalid_request", "user_id or user_key is required");
}
