/**
 * short-id — 基于 CSPRNG + base62 的短 ID 生成器
 *
 * 为什么不用 Math.random().toString(36)：
 *   - Math.random() 是伪随机（V8 xorshift128+），返回 double 尾数只有 52 bit
 *   - 多进程无种子隔离，同一秒多副本启动前几位可能相关
 *   - base36 编码 12 字符看起来占 62 bit 空间，但受限于 52 bit 熵，
 *     单实例 100 万 skill 时碰撞概率约 1.1e-4（生日悖论，非工程可接受值）
 *
 * 本工具用 `crypto.randomBytes` (CSPRNG) + base62 编码：
 *   - base62 = 0-9 A-Z a-z，纯字母数字，无特殊符号（URL/COS/日志友好）
 *   - 12 字符 base62 = 62^12 ≈ 3.23e21 空间 ≈ 71 bit 真熵
 *   - 100 万条时碰撞概率 ≈ 1.5e-10，比现状降低约 73 万倍
 *
 * 用于：
 *   - SkillCore 生成 skill_id（`skl-` + 12 字符 = 16 字符，与现状同长度）
 *   - SqliteSkillStore / TcvdbSkillStore 生成 row_id
 */

import { randomBytes } from "node:crypto";

const BASE62_ALPHABET =
  "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

// 拒绝采样阈值：256 除以 62 向下取整 * 62 = 248
// 字节值 0..247 均匀映射到 62 个字符；248..255 丢弃，避免模偏差。
const REJECTION_THRESHOLD = 248;

/**
 * 生成指定长度的 base62 随机字符串（CSPRNG 熵）。
 *
 * @param len 输出字符数（>=1）
 */
export function randomBase62(len: number): string {
  if (!Number.isInteger(len) || len < 1) {
    throw new RangeError(`randomBase62: len must be a positive integer, got ${len}`);
  }
  let out = "";
  while (out.length < len) {
    // 需要多少字节：估算 (剩余长度 * 256/248) 向上取整 + 少量冗余
    const remaining = len - out.length;
    const need = Math.max(8, Math.ceil((remaining * 256) / REJECTION_THRESHOLD) + 4);
    const buf = randomBytes(need);
    for (let i = 0; i < buf.length && out.length < len; i++) {
      const b = buf[i];
      if (b < REJECTION_THRESHOLD) {
        out += BASE62_ALPHABET[b % 62];
      }
    }
  }
  return out;
}
