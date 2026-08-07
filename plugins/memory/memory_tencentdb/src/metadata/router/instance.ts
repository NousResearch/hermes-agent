/**
 * v3 实例租户：从 x-tdai-service-id 解析 instance_id（v2.8）。
 */
import type { IncomingHttpHeaders } from "node:http";
import { InvalidInstanceIdError, resolveMetadataDbName } from "../store/db-name.js";
import { MetadataError } from "../service/metadata-service.js";

/**
 * 校验 instance_id 可路由到元数据库名；非法时抛 MetadataError。
 * 在路由层统一校验，避免 store 层 InvalidInstanceIdError 泄漏为 500。
 */
export function normalizeInstanceIdForRoute(instanceId: string): string {
  const trimmed = instanceId.trim();
  if (!trimmed) {
    throw new MetadataError("missing_instance_id", "x-tdai-service-id header is required");
  }
  try {
    resolveMetadataDbName(trimmed);
  } catch (err) {
    if (err instanceof InvalidInstanceIdError) {
      throw new MetadataError("invalid_instance_id", err.message);
    }
    throw err;
  }
  return trimmed;
}

export function extractInstanceId(headers: IncomingHttpHeaders): string {
  const raw = headers["x-tdai-service-id"];
  const id = Array.isArray(raw) ? raw[0] : raw ?? "";
  return normalizeInstanceIdForRoute(id);
}
