/**
 * 用户密钥与密码哈希工具。
 *
 * 对应设计文档 §5：
 *   - user_key：`sk-mem-` 前缀 + 24 字节(192bit) base64url 随机段，外部调用鉴权标识
 *   - password：scrypt + 每用户随机盐 + 全局 pepper（不可逆哈希）
 *
 * pepper 由部署时通过环境变量注入（见 loadPasswordHashConfig）。
 * v3.1 起元数据 User 域不再使用 password 哈希；本模块仅保留 generateUserKey 与历史 hash 工具。
 */

import { randomBytes, scryptSync, timingSafeEqual } from "node:crypto";
import { USER_KEY_PREFIX } from "./user-key.js";

/**
 * 生成用户密钥：`sk-mem-` 前缀 + 24 字节(192bit) base64url 随机段。
 *
 * base64url 字符集为 `[A-Za-z0-9_-]`，URL / HTTP Header / JSON 均安全。
 * 192bit 熵远高于碰撞界所需，配合 `key_value` 唯一索引即可保证全局唯一。
 */
export function generateUserKey(): string {
  return USER_KEY_PREFIX + randomBytes(24).toString("base64url");
}

const PASSWORD_CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_";

/** 生成随机密码（默认 12 位，字符集：大小写字母、数字、下划线）。 */
export function generatePassword(length = 12): string {
  const bytes = randomBytes(length);
  let out = "";
  for (let i = 0; i < length; i++) {
    out += PASSWORD_CHARSET[bytes[i]! % PASSWORD_CHARSET.length];
  }
  return out;
}

/** scrypt 参数与存库格式前缀。 */
export const PASSWORD_HASH_PREFIX = "$scrypt$";
const SALT_LEN = 16;
const DEFAULT_SCRYPT_N = 16384;
const DEFAULT_SCRYPT_R = 8;
const DEFAULT_SCRYPT_P = 1;
const DEFAULT_SCRYPT_KEYLEN = 32;
const PEPPER_LEN = 32;

export interface PasswordHashConfig {
  pepper: Buffer;
  scryptN: number;
  scryptR: number;
  scryptP: number;
  keylen: number;
}

/**
 * 从环境变量加载密码哈希配置。
 *
 * 环境变量：
 *   - TDAI_PASSWORD_PEPPER — base64 编码的 32 字节随机值（service 模式必填）
 *   - TDAI_PASSWORD_SCRYPT_N / _R / _P / _KEYLEN — 可选 scrypt 参数
 *
 * @throws 当 pepper 未配置或格式非法时抛异常。
 */
export function loadPasswordHashConfig(env: NodeJS.ProcessEnv = process.env): PasswordHashConfig {
  const pepperB64 = env.TDAI_PASSWORD_PEPPER?.trim();
  if (!pepperB64) {
    throw new Error("TDAI_PASSWORD_PEPPER is not configured (base64-encoded 32-byte secret required)");
  }
  const pepper = Buffer.from(pepperB64, "base64");
  if (pepper.length !== PEPPER_LEN) {
    throw new Error(
      `TDAI_PASSWORD_PEPPER must decode to ${PEPPER_LEN} bytes, got ${pepper.length}`,
    );
  }

  const scryptN = parsePositiveInt(env.TDAI_PASSWORD_SCRYPT_N, DEFAULT_SCRYPT_N);
  const scryptR = parsePositiveInt(env.TDAI_PASSWORD_SCRYPT_R, DEFAULT_SCRYPT_R);
  const scryptP = parsePositiveInt(env.TDAI_PASSWORD_SCRYPT_P, DEFAULT_SCRYPT_P);
  const keylen = parsePositiveInt(env.TDAI_PASSWORD_SCRYPT_KEYLEN, DEFAULT_SCRYPT_KEYLEN);

  return { pepper, scryptN, scryptR, scryptP, keylen };
}

function parsePositiveInt(raw: string | undefined, fallback: number): number {
  if (!raw?.trim()) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || n <= 0 || !Number.isInteger(n)) {
    throw new Error(`Invalid scrypt parameter: ${raw}`);
  }
  return n;
}

function scryptHash(plain: string, salt: Buffer, config: PasswordHashConfig): Buffer {
  const input = Buffer.concat([config.pepper, Buffer.from(plain, "utf8")]);
  return scryptSync(input, salt, config.keylen, {
    N: config.scryptN,
    r: config.scryptR,
    p: config.scryptP,
  });
}

/**
 * 对明文密码做 scrypt+pepper 哈希，返回自描述存库串。
 *
 * 格式：`$scrypt$N,r,p$<salt_b64>$<hash_b64>`
 */
export function hashPassword(plain: string, config: PasswordHashConfig): string {
  const salt = randomBytes(SALT_LEN);
  const hash = scryptHash(plain, salt, config);
  return (
    `${PASSWORD_HASH_PREFIX}${config.scryptN},${config.scryptR},${config.scryptP}` +
    `$${salt.toString("base64url")}$${hash.toString("base64url")}`
  );
}

/** 判断存库串是否为 scrypt 哈希格式。 */
export function isPasswordHash(stored: string): boolean {
  return stored.startsWith(PASSWORD_HASH_PREFIX);
}

/**
 * 验证明文密码与存库哈希是否匹配。格式非法或参数不匹配返回 false（不抛异常）。
 */
export function verifyPasswordHash(
  plain: string,
  stored: string,
  config: PasswordHashConfig,
): boolean {
  try {
    if (!isPasswordHash(stored)) return false;

    const body = stored.slice(PASSWORD_HASH_PREFIX.length);
    const firstSep = body.indexOf("$");
    if (firstSep < 0) return false;

    const params = body.slice(0, firstSep).split(",");
    if (params.length !== 3) return false;
    const [nStr, rStr, pStr] = params;
    const n = Number(nStr);
    const r = Number(rStr);
    const p = Number(pStr);
    if (!Number.isFinite(n) || !Number.isFinite(r) || !Number.isFinite(p)) return false;

    const rest = body.slice(firstSep + 1);
    const secondSep = rest.indexOf("$");
    if (secondSep < 0) return false;

    const salt = Buffer.from(rest.slice(0, secondSep), "base64url");
    const expectedHash = Buffer.from(rest.slice(secondSep + 1), "base64url");

    const verifyConfig: PasswordHashConfig = {
      ...config,
      scryptN: n,
      scryptR: r,
      scryptP: p,
      keylen: expectedHash.length,
    };
    const actualHash = scryptHash(plain, salt, verifyConfig);
    if (actualHash.length !== expectedHash.length) return false;
    return timingSafeEqual(actualHash, expectedHash);
  } catch {
    return false;
  }
}
