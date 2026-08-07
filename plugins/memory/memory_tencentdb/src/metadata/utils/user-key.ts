/**
 * user_key 脱敏与有效性判断（多 key 模型）。
 */

/** user_key 固定前缀，用于识别归属与密钥扫描。 */
export const USER_KEY_PREFIX = "sk-mem-";

/**
 * 通用敏感串脱敏：前 8 字符 + 省略号，不暴露完整 secret。
 * 用于 api-trace 对 password / token / authorization 等泛化密钥的日志脱敏。
 */
export function maskKeyValue(keyValue: string): string {
  if (!keyValue) return "";
  if (keyValue.length <= 8) return `${keyValue}…`;
  return `${keyValue.slice(0, 8)}…`;
}

/**
 * user_key 列表/详情展示脱敏：保留 `sk-mem-` 前缀 + `****` + 末 4 位。
 * 例如 `sk-mem-<32 chars>...e5fG` → `sk-mem-****e5fG`。
 */
export function maskUserKey(keyValue: string): string {
  if (!keyValue) return "";
  const prefix = keyValue.startsWith(USER_KEY_PREFIX) ? USER_KEY_PREFIX : "";
  const body = keyValue.slice(prefix.length);
  const tail = body.length >= 4 ? body.slice(-4) : body;
  return `${prefix}****${tail}`;
}

/** active key 是否已过期（expires_at 为 ISO 字符串）。 */
export function isUserKeyExpired(expiresAt: string | null | undefined, now = Date.now()): boolean {
  if (!expiresAt) return false;
  const t = Date.parse(expiresAt);
  return Number.isFinite(t) && t <= now;
}

export const DEFAULT_MAX_ACTIVE_USER_KEYS = 20;
