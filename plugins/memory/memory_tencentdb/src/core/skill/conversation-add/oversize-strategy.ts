/**
 * SkillOversizeStrategy — 超大兜底切分。
 *
 * 对应设计文档 `2026-07-15-skill-trigger-in-core-design.md` §8。
 *
 * 触发条件：
 *   压缩路径下，压缩后本次 messages + data-current 现有内容仍 > chunkMax。
 *
 * 策略：
 *   从头累加到 headKeepBytes 边界（按 message 边界切分，保证 JSONL 有效）
 *   从尾累加到 tailKeepBytes 边界
 *   中间省略的所有 message → 替换为一条 role=system 的 placeholder 消息
 *
 * 极端情况：单条 message 就超过 head/tail budget → 允许头/尾各装 1 条
 * （保证至少有 head + tail 消息）。
 */

export interface OversizeMessage {
  role: string;
  content: string;
  // 允许透传其它字段
  [key: string]: unknown;
  metadata?: Record<string, unknown>;
}

export interface OversizeOptions {
  chunkMaxBytes: number;
  headKeepBytes: number;
  tailKeepBytes: number;
  /** placeholder 内容模板；`{n}` 替换成省略数，`{bytes}` 替换成省略字节数。 */
  placeholderTemplate: string;
}

export const DEFAULT_OVERSIZE_OPTIONS: OversizeOptions = {
  chunkMaxBytes: 81_920, // 80KB
  headKeepBytes: 20_480, // 20KB
  tailKeepBytes: 20_480, // 20KB
  placeholderTemplate: "[中间 {n} 条消息 / {bytes} 字节内容过长已省略]",
};

export interface OversizeResult {
  /** 处理后的消息序列 */
  messages: OversizeMessage[];
  /** 是否触发了截断（false 表示直接 passthrough） */
  truncated: boolean;
  /** 被省略的 message 条数 */
  omittedMessageCount: number;
  /** 被省略的字节数 */
  omittedBytes: number;
}

function messageBytes(msg: OversizeMessage): number {
  return Buffer.byteLength(JSON.stringify(msg), "utf8");
}

function totalBytes(msgs: OversizeMessage[]): number {
  let sum = 0;
  for (const m of msgs) sum += messageBytes(m);
  return sum;
}

export function applyOversizeStrategy(
  messages: OversizeMessage[],
  optsOverride: Partial<OversizeOptions> = {},
): OversizeResult {
  const opts: OversizeOptions = { ...DEFAULT_OVERSIZE_OPTIONS, ...optsOverride };

  if (messages.length === 0) {
    return { messages: [], truncated: false, omittedMessageCount: 0, omittedBytes: 0 };
  }

  const total = totalBytes(messages);
  if (total <= opts.chunkMaxBytes) {
    return { messages: [...messages], truncated: false, omittedMessageCount: 0, omittedBytes: 0 };
  }

  // 从头累加
  const headMsgs: OversizeMessage[] = [];
  let headBytes = 0;
  let headEnd = 0; // exclusive
  for (let i = 0; i < messages.length; i++) {
    const b = messageBytes(messages[i]!);
    // 允许头至少装 1 条（极端 single message > headKeep 的兜底）
    if (headMsgs.length > 0 && headBytes + b > opts.headKeepBytes) break;
    headMsgs.push(messages[i]!);
    headBytes += b;
    headEnd = i + 1;
    if (headBytes >= opts.headKeepBytes) break;
  }

  // 从尾累加（不要吃回头部区域）
  const tailMsgs: OversizeMessage[] = [];
  let tailBytes = 0;
  let tailStart = messages.length; // inclusive
  for (let i = messages.length - 1; i >= headEnd; i--) {
    const b = messageBytes(messages[i]!);
    if (tailMsgs.length > 0 && tailBytes + b > opts.tailKeepBytes) break;
    tailMsgs.unshift(messages[i]!);
    tailBytes += b;
    tailStart = i;
    if (tailBytes >= opts.tailKeepBytes) break;
  }

  // 头尾之间的省略段
  const omittedSlice = messages.slice(headEnd, tailStart);
  const omittedMessageCount = omittedSlice.length;
  const omittedBytes = totalBytes(omittedSlice);

  // 极端场景：头尾覆盖全部（omitted=0）→ 视为 passthrough
  if (omittedMessageCount === 0) {
    return {
      messages: [...messages],
      truncated: false,
      omittedMessageCount: 0,
      omittedBytes: 0,
    };
  }

  const placeholderContent = opts.placeholderTemplate
    .replace("{n}", String(omittedMessageCount))
    .replace("{bytes}", String(omittedBytes));

  const placeholder: OversizeMessage = {
    role: "system",
    content: placeholderContent,
    metadata: {
      omitted_message_count: omittedMessageCount,
      omitted_bytes: omittedBytes,
    },
  };

  return {
    messages: [...headMsgs, placeholder, ...tailMsgs],
    truncated: true,
    omittedMessageCount,
    omittedBytes,
  };
}
