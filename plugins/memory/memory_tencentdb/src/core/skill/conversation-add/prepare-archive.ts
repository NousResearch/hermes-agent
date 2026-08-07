/**
 * prepareArchivePayload — direct-trigger (`/v3/skill/extract`) 与
 * conversation-add handler 共用的"压缩 + 兜底"归档 payload 组装。
 *
 * 抽出来是为了两条链路**只有一份实现**，不然容易漂移出细微差异。
 *
 * 步骤（来自 `add-handler.ts` handle() 内 §③ 分路径段的等价逻辑）：
 *   ① forceCompress=true → compressMessages(incoming); 否则 passthrough
 *   ② combined = existing + compressed
 *   ③ 仅当 forceCompress=true 且 totalBytes(combined) > chunkMax
 *      → applyOversizeStrategy（兜底截断只发生在压缩路径，与原 add-handler 行为等价）
 *   ④ 返回 { messages, usedCompress, usedOversize }
 *
 * 调用约定：
 *   - conversation/add: existing = data-current, forceCompress = (rawBytes >= threshold)
 *   - skill_extract  : existing = [], forceCompress = true (direct-trigger 恒压缩)
 */

import {
  compressMessages,
  type CompressOptions,
  type CompressibleMessage,
} from "./message-compressor.js";
import {
  applyOversizeStrategy,
  type OversizeMessage,
  type OversizeOptions,
} from "./oversize-strategy.js";

export interface PrepareArchiveOptions {
  compress: CompressOptions;
  oversize: OversizeOptions;
  /** direct-trigger 恒 true；conversation/add 仅压缩路径传 true。 */
  forceCompress: boolean;
}

export interface PrepareArchiveResult {
  messages: OversizeMessage[];
  /** 是否走了 compressMessages（有 tool 消息 > threshold 时才为 true）。 */
  usedCompress: boolean;
  /** 是否触发了 oversize 兜底截断。 */
  usedOversize: boolean;
}

export function prepareArchivePayload(
  existing: OversizeMessage[],
  incoming: CompressibleMessage[],
  opts: PrepareArchiveOptions,
): PrepareArchiveResult {
  // ① forceCompress 决定是否走压缩
  const compressed: CompressibleMessage[] = opts.forceCompress
    ? compressMessages(incoming, opts.compress)
    : incoming;

  // usedCompress 反映"是否真的有内容被压掉"——不是"是否 forceCompress"。
  // compressMessages 只在 tool 消息 content > threshold 时才实际改内容，
  // 短消息即便 forceCompress 也 identity-return，不算真压缩。
  const usedCompress = opts.forceCompress && compressed.some(
    (m, i) => m !== incoming[i],
  );

  // ② 拼接
  const combined: OversizeMessage[] = [
    ...existing,
    ...(compressed as unknown as OversizeMessage[]),
  ];

  // ③ 判断是否需要 oversize 兜底 —— 仅在 forceCompress 路径下触发，跟原
  //    add-handler 行为对齐 (常规路径下不会 combined > chunkMax; 强压缩后仍超
  //    才走兜底)。skill_extract 侧 forceCompress=true, 该判定自然生效。
  if (!opts.forceCompress) {
    return { messages: combined, usedCompress, usedOversize: false };
  }
  const combinedBytes = totalMessagesBytes(combined);
  if (combinedBytes <= opts.oversize.chunkMaxBytes) {
    return { messages: combined, usedCompress, usedOversize: false };
  }

  const out = applyOversizeStrategy(combined, opts.oversize);
  return {
    messages: out.messages,
    usedCompress,
    usedOversize: out.truncated,
  };
}

function totalMessagesBytes(msgs: Array<{ role: string; content: string }>): number {
  let sum = 0;
  for (const m of msgs) sum += Buffer.byteLength(JSON.stringify(m), "utf8");
  return sum;
}
