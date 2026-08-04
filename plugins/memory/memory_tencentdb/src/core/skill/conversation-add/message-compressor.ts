/**
 * SkillMessageCompressor — compress oversized tool_call / tool_result payloads
 * before they enter the skill buffer.
 *
 * 对应设计文档 `2026-07-15-skill-trigger-in-core-design.md` §6：
 *   - 只压 tool_call / tool_result；user / assistant / system 永不压缩
 *   - content 字节数 > 2KB 才压缩，头 1KB + 尾 1KB + 中间占位提示
 *   - metadata.truncated = true, metadata.original_bytes = <原始字节数>
 *
 * 字节切分实现说明：
 *   直接对 Buffer 做 slice 可能截到 UTF-8 多字节字符的中间，转回 string 会
 *   出现 U+FFFD 替换字符。这里我们对 Buffer 做切片、再 toString('utf8')，
 *   Node 侧会把结尾/开头不完整的多字节序列替换成 U+FFFD——但整体不影响
 *   下游 LLM review 的语义（提示语里说明了截断）。对于 tool payload 这种
 *   通常是 ASCII/JSON 的场景，边界字符损坏概率极低；测试对齐宽松断言。
 */

export type CompressibleRole =
  | "user"
  | "assistant"
  | "tool_call"
  | "tool_result"
  | "system";

export interface CompressibleMessage {
  role: CompressibleRole;
  content: string;
  /** Optional tool identity for tool_call / tool_result. */
  tool_name?: string;
  tool_call_id?: string;
  timestamp?: number;
  metadata?: Record<string, unknown>;
}

export interface CompressOptions {
  /** 字节阈值；tool 消息 content bytes > 阈值才压缩。默认 2048 (2KB)。 */
  toolContentThresholdBytes: number;
  /** 头部保留字节数。默认 1024 (1KB)。 */
  headBytes: number;
  /** 尾部保留字节数。默认 1024 (1KB)。 */
  tailBytes: number;
  /** 中间占位字符串。 */
  placeholder: string;
}

export const DEFAULT_COMPRESS_OPTIONS: CompressOptions = {
  toolContentThresholdBytes: 2048,
  headBytes: 1024,
  tailBytes: 1024,
  placeholder: "\n\n[中间内容过长已被压缩，只展示头尾]\n\n",
};

const COMPRESSIBLE_ROLES = new Set<CompressibleRole>(["tool_call", "tool_result"]);

/**
 * Compress a single message. Returns a new object if compressed, otherwise
 * returns the original message reference unchanged.
 */
export function compressMessage(
  msg: CompressibleMessage,
  optsOverride: Partial<CompressOptions> = {},
): CompressibleMessage {
  const opts: CompressOptions = { ...DEFAULT_COMPRESS_OPTIONS, ...optsOverride };
  if (!COMPRESSIBLE_ROLES.has(msg.role)) return msg;

  const bytes = Buffer.byteLength(msg.content, "utf8");
  if (bytes <= opts.toolContentThresholdBytes) return msg;

  const buf = Buffer.from(msg.content, "utf8");
  const head = buf.subarray(0, opts.headBytes).toString("utf8");
  const tail = buf.subarray(buf.length - opts.tailBytes).toString("utf8");

  const nextContent = `${head}${opts.placeholder}${tail}`;
  const nextMetadata: Record<string, unknown> = {
    ...(msg.metadata ?? {}),
    truncated: true,
    original_bytes: bytes,
  };

  return {
    ...msg,
    content: nextContent,
    metadata: nextMetadata,
  };
}

/**
 * Compress an array of messages. Returns a new array; unchanged messages
 * share the original reference (identity-preserving for downstream diffing).
 */
export function compressMessages(
  messages: CompressibleMessage[],
  optsOverride: Partial<CompressOptions> = {},
): CompressibleMessage[] {
  if (messages.length === 0) return [];
  return messages.map((m) => compressMessage(m, optsOverride));
}
