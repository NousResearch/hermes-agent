/**
 * SkillConversationAddHandler — §7 Handler 主流程。
 *
 * 处理 `POST /v3/skill/conversation/add`：
 *   ① 校验必填字段 + role
 *   ② 计算 raw_bytes
 *   ③ 分路径：normal (< requestCompressThreshold) / compressed (≥) / oversize (拼接后 > chunkMax)
 *   ④ 拼接 data-current，累加计数
 *   ⑤ 判阈值 → 触发归档 (SkillTriggerService)
 *   ⑥ 写回 data-current + meta
 */

import {
  DEFAULT_COMPRESS_OPTIONS,
  type CompressibleMessage,
  type CompressOptions,
  type CompressibleRole,
} from "./message-compressor.js";
import {
  DEFAULT_OVERSIZE_OPTIONS,
  type OversizeMessage,
  type OversizeOptions,
} from "./oversize-strategy.js";
import { prepareArchivePayload } from "./prepare-archive.js";
import type { SkillBufferStorage, SessionKey, SessionMeta } from "./buffer-storage.js";
import type { SkillTriggerService } from "./trigger-service.js";
import { obsLogger } from "../../report/obs-logger.js";

const VALID_ROLES: ReadonlySet<CompressibleRole> = new Set([
  "user",
  "assistant",
  "tool_call",
  "tool_result",
  "system",
]);

/**
 * 归档阈值 `tool_call_count` 的计数集合。
 *
 * **只算 `tool_call`, 不算 `tool_result`。** 二者天然 1:1 配对 (每次
 * agent 调工具都会带回一次 result), 把两者都算等于把计数翻倍, 用户会
 * 观察到"agent 调 5 次工具就归档"——完全不是配置里的 10。
 *
 * 具体来说, VALID_ROLES 里 "tool_call" 是 agent 主动发起的调用,
 * "tool_result" 是配对的返回。归档触发的语义是"agent 用工具的次数",
 * 所以只数 call 一侧。
 *
 * 校验路径 (validate() 里) 依然对 tool_call 和 tool_result 都要求
 * tool_call_id —— 那是**结构合法性**校验, 跟计数无关, 两码事。
 */
const TOOL_CALL_ROLES: ReadonlySet<CompressibleRole> = new Set(["tool_call"]);

/** 校验时需要 tool_call_id 的 role 集合 (call 和 result 都要携带配对锚点)。 */
const TOOL_PAIR_ROLES: ReadonlySet<CompressibleRole> = new Set(["tool_call", "tool_result"]);

const ID_FORBIDDEN_CHAR = "|";

export interface AddConversationInput {
  session_id: string;
  space_id: string;
  user_id: string;
  team_id: string;
  agent_id: string;
  /** 业务侧 task 引用，透传到 archive 落地时的 task.task_ref_id。 */
  task_id?: string;
  messages: CompressibleMessage[];
  /**
   * 上游 HTTP handler 的 req_id，用于 obsLogger 分段事件关联链路。
   * 缺省则事件字段少一个 req_id，业务逻辑不受影响。
   */
  perfRequestId?: string;
}

export interface AddConversationResult {
  /** 语义状态：ok=正常追加 / archived=触发了归档。 */
  status: "ok" | "archived";
  archived?: {
    task_id: string;
    archived_at_ms: number;
    archive_key: string;
    /** normal 达阈值触发 / compressed 必触发 / oversize 兜底后触发 */
    reason: "tool_calls" | "bytes" | "compressed" | "oversize";
  };
}

export interface HandlerThresholds {
  /** tool_call 累计阈值。默认 10。 */
  toolCallThreshold: number;
  /** 字节累计阈值。默认 40960 (40KB)。 */
  bytesThreshold: number;
  /** 本次 add 字节 ≥ 此值走压缩路径。默认 40960。 */
  requestCompressThresholdBytes: number;
}

export const DEFAULT_HANDLER_THRESHOLDS: HandlerThresholds = {
  toolCallThreshold: 10,
  bytesThreshold: 40 * 1024,
  requestCompressThresholdBytes: 40 * 1024,
};

export interface SkillConversationAddHandlerOptions {
  buffer: SkillBufferStorage;
  trigger: SkillTriggerService;
  thresholds?: Partial<HandlerThresholds>;
  compressOptions?: Partial<CompressOptions>;
  oversizeOptions?: Partial<OversizeOptions>;
  now?: () => number;
}

export class HandlerValidationError extends Error {
  constructor(public readonly field: string, message: string) {
    super(message);
    this.name = "HandlerValidationError";
  }
}

export class SkillConversationAddHandler {
  private readonly buffer: SkillBufferStorage;
  private readonly trigger: SkillTriggerService;
  private readonly thresholds: HandlerThresholds;
  private readonly compressOptions: CompressOptions;
  private readonly oversizeOptions: OversizeOptions;
  private readonly now: () => number;

  constructor(opts: SkillConversationAddHandlerOptions) {
    this.buffer = opts.buffer;
    this.trigger = opts.trigger;
    this.thresholds = { ...DEFAULT_HANDLER_THRESHOLDS, ...opts.thresholds };
    this.compressOptions = { ...DEFAULT_COMPRESS_OPTIONS, ...opts.compressOptions };
    this.oversizeOptions = { ...DEFAULT_OVERSIZE_OPTIONS, ...opts.oversizeOptions };
    this.now = opts.now ?? (() => Date.now());
  }

  async handle(input: AddConversationInput): Promise<AddConversationResult> {
    // [obs] handler 内部分段：readBuffer / prepareArchive / trigger.archive / writeBack。
    // 走 obsLogger 底座（结构化事件 + FileLogger + ClickHouse 后端），
    // 通过 req_id 与上游 handleConversationAdd + trigger + worker 关联全链路。
    const rid = input.perfRequestId;

    // ① 校验
    this.validate(input);
    const sess: SessionKey = {
      space_id: input.space_id,
      user_id: input.user_id,
      team_id: input.team_id,
      agent_id: input.agent_id,
      session_id: input.session_id,
    };

    // ② 计算 raw_bytes
    const rawBytes = totalMessagesBytes(input.messages);

    // ③ 分路径：读现状 + 走共享 helper 做压缩 + 兜底
    const useCompress = rawBytes >= this.thresholds.requestCompressThresholdBytes;
    const t0Buf = Date.now();
    const [current, meta] = await Promise.all([
      this.buffer.readCurrent(sess),
      this.buffer.readMeta(sess),
    ]);
    obsLogger.info("skill.add_handler.read_buffer", {
      req_id: rid, session_id: input.session_id,
      dur_ms: Date.now() - t0Buf,
      current_msgs: current.messages.length,
      raw_bytes: rawBytes,
      use_compress: useCompress,
    });

    // conversation-add 特有语义：只有压缩路径才走 oversize 兜底 (原实现见下方注释);
    // 用 helper 时，forceCompress=useCompress，当 useCompress=false 时 helper 内部
    // 也不会走 applyOversizeStrategy——因为常规路径下 combinedBytes 不该 > chunkMax
    // (那种情况下 rawBytes 早已 >= requestCompressThresholdBytes 走了压缩路径)。
    // helper 里的 oversize 判定跟原实现语义等价：都是"combined > chunkMax"。
    const t0Prep = Date.now();
    const prepared = prepareArchivePayload(
      current.messages as OversizeMessage[],
      input.messages,
      {
        compress: this.compressOptions,
        oversize: this.oversizeOptions,
        forceCompress: useCompress,
      },
    );
    obsLogger.info("skill.add_handler.prepare_archive", {
      req_id: rid, session_id: input.session_id,
      dur_ms: Date.now() - t0Prep,
      msg_in: input.messages.length,
      msg_out: prepared.messages.length,
      used_oversize: prepared.usedOversize,
    });
    const combinedMessages: OversizeMessage[] = prepared.messages;
    const usedOversize = prepared.usedOversize;

    // ④ 更新 meta 计数
    // 只数 tool_call, 不数 tool_result —— 二者 1:1 配对, 数两遍会让阈值 10
    // 变成实际"5 次工具调用即归档", 违背配置语义。详见 TOOL_CALL_ROLES 注释。
    const addedToolCalls = countRoles(input.messages, TOOL_CALL_ROLES);
    const nextTool = meta.tool_call_count + addedToolCalls;
    const nextBytes = meta.byte_count + rawBytes;

    // ⑤ 阈值判定
    const hitTool = nextTool >= this.thresholds.toolCallThreshold;
    const hitBytes = nextBytes >= this.thresholds.bytesThreshold;
    const shouldArchive = useCompress || hitTool || hitBytes;

    let result: AddConversationResult = { status: "ok" };

    if (shouldArchive) {
      // 归档段
      const reason: NonNullable<AddConversationResult["archived"]>["reason"] = usedOversize
        ? "oversize"
        : useCompress
          ? "compressed"
          : hitTool
            ? "tool_calls"
            : "bytes";

      const t0Arch = Date.now();
      const archiveRes = await this.trigger.archive({
        session: sess,
        bufferAtTrigger: { messages: combinedMessages as Array<Record<string, unknown>> },
        taskRefId: input.task_id,
        // 透传 req_id 给 trigger 内部分段事件（write_archive / mutex_* / enqueue_agent）
        perfRequestId: input.perfRequestId,
      });
      obsLogger.info("skill.add_handler.trigger_archive", {
        req_id: rid, session_id: input.session_id,
        dur_ms: Date.now() - t0Arch,
        task_id: archiveRes.taskId,
        archive_key: archiveRes.archiveKey,
        reason,
      });

      // 归档后清空 data-current + 计数
      const nowMs = this.now();
      const nextMeta: SessionMeta = {
        session_id: sess.session_id,
        space_id: sess.space_id,
        user_id: sess.user_id,
        team_id: sess.team_id,
        agent_id: sess.agent_id,
        tool_call_count: 0,
        byte_count: 0,
        last_appended_at_ms: nowMs,
        last_archived_at_ms: archiveRes.archivedAtMs,
      };

      const t0Wb = Date.now();
      await Promise.all([
        this.buffer.writeCurrent(sess, { messages: [] }),
        this.buffer.writeMeta(sess, nextMeta),
      ]);
      obsLogger.info("skill.add_handler.write_back", {
        req_id: rid, session_id: input.session_id,
        dur_ms: Date.now() - t0Wb,
        archived: true,
      });

      result = {
        status: "archived",
        archived: {
          task_id: archiveRes.taskId,
          archived_at_ms: archiveRes.archivedAtMs,
          archive_key: archiveRes.archiveKey,
          reason,
        },
      };
    } else {
      // 未触发归档：直接把拼接后的 data-current 写回
      const nowMs = this.now();
      const nextMeta: SessionMeta = {
        session_id: sess.session_id,
        space_id: sess.space_id,
        user_id: sess.user_id,
        team_id: sess.team_id,
        agent_id: sess.agent_id,
        tool_call_count: nextTool,
        byte_count: nextBytes,
        last_appended_at_ms: nowMs,
        last_archived_at_ms: meta.last_archived_at_ms,
      };
      const t0Wb = Date.now();
      await Promise.all([
        this.buffer.writeCurrent(sess, { messages: combinedMessages as Array<Record<string, unknown>> }),
        this.buffer.writeMeta(sess, nextMeta),
      ]);
      obsLogger.info("skill.add_handler.write_back", {
        req_id: rid, session_id: input.session_id,
        dur_ms: Date.now() - t0Wb,
        archived: false,
        tool_count: nextTool,
        byte_count: nextBytes,
      });
    }

    return result;
  }

  private validate(input: AddConversationInput): void {
    const required: Array<keyof AddConversationInput> = [
      "session_id",
      "space_id",
      "user_id",
      "team_id",
      "agent_id",
    ];
    for (const f of required) {
      const v = input[f];
      if (typeof v !== "string" || v.length === 0) {
        throw new HandlerValidationError(String(f), `${String(f)} is required and must be non-empty string`);
      }
      if ((v as string).includes(ID_FORBIDDEN_CHAR)) {
        throw new HandlerValidationError(
          String(f),
          `${String(f)} must not contain '|' (reserved for agent tuple)`,
        );
      }
    }
    if (!Array.isArray(input.messages) || input.messages.length === 0) {
      throw new HandlerValidationError("messages", "messages must be a non-empty array");
    }
    for (let i = 0; i < input.messages.length; i++) {
      const m = input.messages[i]!;
      if (!VALID_ROLES.has(m.role as CompressibleRole)) {
        throw new HandlerValidationError(`messages[${i}].role`, `invalid role: ${m.role}`);
      }
      if (typeof m.content !== "string") {
        throw new HandlerValidationError(`messages[${i}].content`, "content must be string");
      }
      if (TOOL_PAIR_ROLES.has(m.role as CompressibleRole)) {
        // tool_call_id 是**必须**的（tool_call 和 tool_result 通过它配对）
        // tool_name 是**可选**的：Anthropic 协议 tool_use block 里有 name, OpenAI 协议
        //   role=tool 消息本身没有 tool_name 字段, 只有 tool_call_id。要求 tool_name
        //   必填会让 proxy 侧被迫反查 assistant.tool_calls 才能填, 属于协议差异导致
        //   的绕圈；干脆放宽为 optional (对 skill 抽取而言, content 才是关键)。
        if (typeof m.tool_call_id !== "string" || m.tool_call_id.length === 0) {
          throw new HandlerValidationError(
            `messages[${i}].tool_call_id`,
            "tool_call/tool_result must carry tool_call_id",
          );
        }
        if (m.tool_name !== undefined && (typeof m.tool_name !== "string" || m.tool_name.length === 0)) {
          throw new HandlerValidationError(
            `messages[${i}].tool_name`,
            "tool_name if provided must be non-empty string",
          );
        }
      }
    }
  }
}

function totalMessagesBytes(msgs: CompressibleMessage[]): number {
  let sum = 0;
  for (const m of msgs) {
    sum += Buffer.byteLength(JSON.stringify(m), "utf8");
  }
  return sum;
}

function countRoles(msgs: CompressibleMessage[], roles: ReadonlySet<CompressibleRole>): number {
  let n = 0;
  for (const m of msgs) if (roles.has(m.role as CompressibleRole)) n++;
  return n;
}
