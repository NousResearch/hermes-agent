/**
 * 带业务前缀的唯一 ID 生成。
 *
 * 对应设计文档 §4 ID 生成规范（v3.1）：
 *   - 主体/资源实体：`{prefix}-{4位时间戳Base36}{6位随机Base36}`（§4.4 方案 A）
 *   - 关联关系表：UUID v4（`randomUUID()`）
 */

import { randomUUID } from "node:crypto";

const CHARS = "abcdefghijklmnopqrstuvwxyz0123456789";
const BASE = CHARS.length; // 36
const TS_LEN = 4;
const RAND_LEN = 6;

/** @deprecated v3.1 关联表改 UUID；保留常量供旧测试迁移参考。 */
export const RELATION_ID_LEN = 36;

/** 业务实体 ID 前缀映射（关联表无前缀）。 */
export const ID_PREFIX = {
  user: "usr",
  team: "team",
  agent: "agt",
  task: "task",
  asset: "ast",
  userKey: "uky",
} as const;

export type IdPrefix = (typeof ID_PREFIX)[keyof typeof ID_PREFIX];

/** 把非负整数编码为定长 Base36（不足左侧补 0）。 */
function encodeBase36(value: number, length: number): string {
  let out = "";
  let remaining = value;
  for (let i = 0; i < length; i++) {
    out = CHARS[remaining % BASE] + out;
    remaining = Math.floor(remaining / BASE);
  }
  return out;
}

/** 生成 length 位随机 Base36 串。 */
function randomBase36(length: number): string {
  let out = "";
  for (let i = 0; i < length; i++) {
    out += CHARS[Math.floor(Math.random() * BASE)];
  }
  return out;
}

/**
 * 生成带前缀的实体 ID，如 `usr-3mfxa3b9c1`。
 *
 * @param prefix 业务前缀（见 ID_PREFIX）
 */
export function generateId(prefix: string): string {
  const ts = Math.floor(Date.now() / 1000) % BASE ** TS_LEN;
  const tsPart = encodeBase36(ts, TS_LEN);
  const randPart = randomBase36(RAND_LEN);
  return `${prefix}-${tsPart}${randPart}`;
}

/** 生成关联表主键（UUID v4）。 */
export function generateRelationId(): string {
  return randomUUID();
}

/**
 * 校验 ID 合法性。
 *
 * @param id 待校验 ID
 * @param prefix 可选，指定时要求 ID 以 `{prefix}-` 开头；不指定时仅校验非空
 *               （兼容存量 ULID 格式）。
 */
export function isValidId(id: string, prefix?: string): boolean {
  if (!id || typeof id !== "string") return false;
  if (prefix === undefined) return id.length > 0;
  return id.startsWith(`${prefix}-`);
}
