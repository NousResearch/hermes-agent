/**
 * v3 元数据 API 鉴权中间件（v3.1）。
 *
 * Layer 1（网关）：`Authorization: Bearer <KERNEL_AUTH_TOKEN>` — 在 server.ts checkAuthForV2 校验。
 * Layer 3（用户身份）：`x-tdai-user-key` — 本模块解析为 userId / isSystemAdmin。
 *
 * `/v3/meta/*` 除 `auth/verify` 外均须 user-key。
 * 无 user-key 的运维接口见 `/v3/internal/meta/*`。
 */

import type { IncomingHttpHeaders } from "node:http";
import type { MetadataService } from "../service/metadata-service.js";

export interface V3AuthContext {
  /** 原始 user_key（来自 x-tdai-user-key）。 */
  token: string;
  /** 解析出的用户 ID（合法 user_key 时）。 */
  userId?: string;
  /** 历史字段：/v3/meta/* 上不再授予 bootstrap isAdmin。 */
  isAdmin: boolean;
  /** user_key 对应 user_type === system_admin。 */
  isSystemAdmin: boolean;
}

export interface V3AuthResult {
  ok: boolean;
  status?: number;
  reason?: string;
  ctx?: V3AuthContext;
}

/** 公开接口中可不传 x-tdai-user-key 的路径（仍须 Bearer + x-tdai-service-id）。 */
export const V3_NO_USER_KEY_ROUTES = new Set([
  "/v3/meta/auth/verify",
]);

/** 从 x-tdai-user-key 头提取用户 API 密钥。 */
export function extractUserKeyHeader(headers: IncomingHttpHeaders): string {
  const raw = headers["x-tdai-user-key"];
  const h = Array.isArray(raw) ? raw[0] : (raw ?? "");
  return h.trim();
}

/**
 * 解析 user_key → 用户上下文。空 key 调用方不应传入（由路由层处理 bootstrap）。
 */
export async function authenticateV3(
  userKey: string,
  service: MetadataService,
): Promise<V3AuthResult> {
  if (!userKey) {
    return { ok: false, status: 401, reason: "missing_user_key" };
  }

  if (service.isConfiguredMemorySystemUserKey(userKey)) {
    return { ok: false, status: 401, reason: "invalid_user_key" };
  }

  const user = await service.verifyAuth(userKey);
  if (!user) {
    return { ok: false, status: 401, reason: "invalid_user_key" };
  }

  const isSystemAdmin = user.user_type === "system_admin";
  return {
    ok: true,
    ctx: { token: userKey, userId: user.user_id, isAdmin: false, isSystemAdmin },
  };
}
