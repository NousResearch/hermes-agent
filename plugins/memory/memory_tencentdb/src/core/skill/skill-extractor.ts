/**
 * SkillExtractor — 接受结构化 ExtractMessage[] 的抽取入口
 *
 * 与旧 SkillExtractor 的差异：
 *   - 入参 messages 必须是 `ExtractMessage[]`，不再接受裸字符串
 *   - 内部把 messages 串成 transcript（保留 role 标记）
 *   - 工具调用走 SkillToolsV2（操作 SkillCore）
 *   - 候选返回 ExtractedSkillCandidate 形态（含 skill_id / version）
 *
 * 每次调用都走 LLM，不做任何对话级去重/缓存 —— 缓存机制已移除。
 */

import type { ExtractMessage } from "./types.js";
import type { SkillCore } from "./skill-core.js";
import { createSkillTools, type ExtractedSkillCandidate } from "./skill-tools.js";
import type {
  ISkillExtractor,
  ConversationMessage,
  ExtractedCandidate,
} from "./queue/types.js";
import { metricProducer } from "../report/kafka-metric-producer.js";
import { trace } from "../report/trace.js";
import { obsLogger } from "../report/obs-logger.js";

const TAG = "[skill-extractor]";

export interface ExtractorRunner {
  run(params: {
    prompt: string;
    systemPrompt?: string;
    tools?: Record<string, unknown>;
    enableTools?: boolean;
    maxIterations?: number;
    /** Output token 上限；不填由 runner 兜底 (标准 runner 走 llm.maxTokens)。 */
    maxTokens?: number;
    taskId: string;
    timeoutMs?: number;
    signal?: AbortSignal;
    /** Langfuse trace name（用于 UI 筛选与命名，见 core/types.ts LLMRunParams）。 */
    traceName?: string;
    /** Langfuse tags（用于 UI 筛选）。 */
    tags?: string[];
    /** Langfuse 顶级 sessionId。 */
    sessionId?: string;
    /** Langfuse 顶级 userId。 */
    userId?: string;
    /** 实例 id；透传成 telemetry 的 instanceId（缺失时 runner 侧兜底 "unknown"）。 */
    instanceId?: string;
  }): Promise<string>;
}

export interface ExtractorOptions {
  core: SkillCore;
  runner?: ExtractorRunner;
  systemPrompt?: string;
  maxIterations?: number;
  /** Transcript head-tail truncation: chars to keep from the start (default 8000). */
  headChars?: number;
  /** Transcript head-tail truncation: chars to keep from the end (default 32000). */
  tailChars?: number;
  /**
   * Skill review 单次 LLM 调用输出 token 上限。不填 → 由 runner 继承 llm.maxTokens。
   * 与 llm.maxTokens 独立配置：skill review 输出（含 tool-call 参数里的 SKILL.md
   * 全文）通常比其它 stage 大，可能需要单独调高。
   */
  maxTokens?: number;
  logger?: { info(msg: string): void; warn(msg: string): void; error(msg: string): void };
}

export interface ExtractInput {
  user_id: string;
  team_id: string;
  agent_id: string;
  task_id?: string;
  session_id?: string;
  /** 实例 id（= space_id）；透传成 runner telemetry 的 instanceId。 */
  space_id?: string;
  messages: ExtractMessage[];
  options?: {
    max_iterations?: number;
  };
  /** 主 Agent 的抽取提示，有值时注入到抽取 LLM 的 user prompt 最前面。 */
  reason?: string;
}

export interface ExtractResult {
  candidates: ExtractedSkillCandidate[];
  text?: string;
}

export class SkillExtractor {
  private readonly core: SkillCore;
  private readonly runner?: ExtractorRunner;
  private readonly systemPrompt: string;
  private readonly maxIterations: number;
  private readonly headChars: number;
  private readonly tailChars: number;
  private readonly maxTokens?: number;
  private readonly logger?: ExtractorOptions["logger"];

  constructor(opts: ExtractorOptions) {
    this.core = opts.core;
    this.runner = opts.runner;
    this.systemPrompt = opts.systemPrompt ?? "You are a Skill Review Agent. Use tools to look at existing skills, decide what to add/improve, and call skill_create / skill_update / skill_patch / skill_files_write to persist.";
    this.maxIterations = opts.maxIterations ?? 16;
    this.headChars = opts.headChars ?? 8000;
    this.tailChars = opts.tailChars ?? 32000;
    this.maxTokens = opts.maxTokens;
    this.logger = opts.logger;
  }

  async extract(input: ExtractInput): Promise<ExtractResult> {
    const { messages } = input;
    if (!Array.isArray(messages) || messages.length === 0) {
      throw new Error("ExtractV2: messages must be a non-empty array of ExtractMessage");
    }
    // [obs] 一次抽取一条汇总事件；LLM 内部 iteration 走 langfuse trace（trace.report
    // → OTel Span → Langfuse SpanProcessor 过滤上报），这里只做「入口→出口」
    // 的耗时 / 候选数量汇总，用 task_id 做 anchor 跟 worker 段对齐。obsLogger
    // 内部 try/catch + FileLogger + 后端降级，logger 挂了也不影响抽取本身。
    const t0 = Date.now();
    const transcript = formatTranscript(messages);

    const truncated = truncateHeadTail(transcript, this.headChars, this.tailChars);
    const recentBlock = await this.buildRecentSkillsBlock(input);
    let prompt = recentBlock ? `${recentBlock}\n\n---\n\n${truncated}` : truncated;

    // 主 Agent 注入抽取提示（reason 非空时放在 prompt 最前面）
    if (input.reason && input.reason.trim().length > 0) {
      const hintBlock = [
        "## 主 Agent 的抽取提示",
        "以下是主 Agent 对本次对话的说明，请重点参考其意图进行抽取：",
        input.reason,
      ].join("\n");
      prompt = `${hintBlock}\n\n---\n\n${prompt}`;
    }

    const auditSink: ExtractedSkillCandidate[] = [];

    if (!this.runner) {
      // No runner injected (test environment / disabled) → 返回空候选
      this.logger?.info(`${TAG} no runner provided; returning empty candidates`);
      obsLogger.info("skill.extractor.extract", {
        task_id: input.task_id,
        dur_ms: Date.now() - t0,
        msg_count: messages.length,
        candidates: 0,
        skipped: "no_runner",
      });
      return { candidates: [] };
    }

    const tools = createSkillTools({
      core: this.core,
      user_id: input.user_id,
      team_id: input.team_id,
      agent_id: input.agent_id,
      task_id: input.task_id,
      auditSink,
      logger: this.logger,
    });

    let text: string;
    try {
      text = await this.runner.run({
        prompt,
        systemPrompt: this.systemPrompt,
        tools,
        enableTools: true,
        maxIterations: input.options?.max_iterations ?? this.maxIterations,
        maxTokens: this.maxTokens,
        taskId: `skill-extract-${input.task_id ?? "unknown"}`,
        // Langfuse trace 语义：让此次抽取在 Langfuse UI 有稳定 name / 可筛选 tags。
        // 详见 core/types.ts LLMRunParams 的 traceName/tags/sessionId/userId 注释。
        traceName: "skill.extract",
        tags: [
          "skill-extract",
          `team:${input.team_id}`,
          `agent:${input.agent_id}`,
        ],
        sessionId: input.session_id,
        userId: input.user_id,
        instanceId: input.space_id,
      });
    } catch (e) {
      // 一条 warn 汇总失败，包含 task_id / err_name / dur —— Worker 侧再按分类
      // (transient / permanent) 决定 requeue 还是 DLQ。这里不吞异常。
      const dur = Date.now() - t0;
      obsLogger.warn("skill.extractor.extract", {
        task_id: input.task_id,
        dur_ms: dur,
        msg_count: messages.length,
        candidates: auditSink.length,
        err_name: (e as Error).name,
        err_msg: (e as Error).message,
      });
      try {
        trace.report("skill.extractor.extract", {
          task_id: input.task_id,
          team_id: input.team_id,
          agent_id: input.agent_id,
          session_id: input.session_id,
          msg_count: messages.length,
          candidates: auditSink.length,
          dur_ms: dur,
          success: false,
          error: (e as Error).message,
        });
      } catch { /* noop */ }
      throw e;
    }

    try { metricProducer.send({ metric: "skill.extract.candidates", instanceId: input.team_id, value: auditSink.length }); } catch { /* noop */ }

    const dur = Date.now() - t0;
    obsLogger.info("skill.extractor.extract", {
      task_id: input.task_id,
      dur_ms: dur,
      msg_count: messages.length,
      candidates: auditSink.length,
      prompt_chars: prompt.length,
    });
    try {
      trace.report("skill.extractor.extract", {
        task_id: input.task_id,
        team_id: input.team_id,
        agent_id: input.agent_id,
        session_id: input.session_id,
        msg_count: messages.length,
        candidates: auditSink.length,
        prompt_chars: prompt.length,
        dur_ms: dur,
        success: true,
      });
    } catch { /* noop */ }

    return { candidates: auditSink, text };
  }

  /**
   * 取该 agent（team_id + agent_id 维度）最近触达的 ≤5 个技能，拼成一段前缀注入
   * user prompt。提供 skill_list 给不了的「时间先验」——在累积快照场景里，模型据此
   * 更容易意识到「这是我刚从本会话建的技能，应 update / no-op 而非重复 create」。
   *
   * 定位是「最近上下文、非穷举」，不替代 skill_list；按 updated_at 倒序（store 已排序）。
   * best-effort：失败只告警，不影响抽取。
   */
  private async buildRecentSkillsBlock(input: ExtractInput): Promise<string> {
    try {
      const { items } = await this.core.list({
        user_id: input.user_id,
        team_id: input.team_id,
        agent_id: input.agent_id,
        pagination: { limit: 5 },
      });
      if (!items.length) return "";
      const lines = items.map((s) => {
        const desc = (s.description ?? "").trim().replace(/\s+/g, " ");
        const short = desc.length > 100 ? `${desc.slice(0, 100)}…` : desc;
        return short ? `- ${s.name} — ${short}` : `- ${s.name}`;
      });
      return [
        "## Skills you (this agent) recently wrote",
        "Most recent first; not exhaustive — still call skill_list before deciding.",
        ...lines,
      ].join("\n");
    } catch (e) {
      this.logger?.warn(`${TAG} buildRecentSkillsBlock failed: ${(e as Error).message}`);
      return "";
    }
  }
}

// ═════════════════════════════════════════════════════════════════════
//  helpers
// ═════════════════════════════════════════════════════════════════════

/**
 * 把 ExtractMessage[] 序列化成给 Skill Review Agent 看的 transcript。
 *
 * 关键设计（对齐 SKILL_REVIEW_PROMPT 的 "Role isolation" 段）：
 *   - role 前缀用非自然的 `<<past-xxx>>` 双尖括号 tag，而不是 `[user]` / `[assistant]`。
 *     后者是 chat completion 的原生 role signal，模型会本能把 transcript 尾部的
 *     `[assistant]` 当成"该我续写下一 turn"的锚点，直接接着写主 agent 的回复。
 *     `<<past-user>>` 这类非典型 tag 打破这个暗示，让模型明确知道"这些是我在
 *     review 的历史内容，不是我要扮演的角色"。
 *   - 末尾追加 `<<end-of-transcript>>` 锚点 + 一句"现在按 system prompt 里的
 *     output contract 决策"。没有这个锚点时，模型看到 transcript 直接结束在
 *     一段 assistant 长回复上，很容易走"再续写一段类似风格"的路径。
 *
 * 相关背景：docs/2026-07-28 skill extractor role-capture 分析（trace
 * f546ab8c-c7c5-4598-a310-6b2162372e7c，抽取 LLM 完全忽略 SKILL_REVIEW_PROMPT
 * 契约，产出 1235 tokens 的主对话续写，零 tool call、零 "Nothing to save."）。
 */
function formatTranscript(messages: ExtractMessage[]): string {
  const body = messages.map((m) => `<<past-${m.role}>>\n${m.content}`).join("\n\n");
  return `${body}\n\n<<end-of-transcript>>\nAbove is the past conversation to review. Now decide, and respond only per the output contract in the system prompt.`;
}

function truncateHeadTail(s: string, head: number, tail: number): string {
  if (s.length <= head + tail) return s;
  return `${s.slice(0, head)}\n\n... [truncated ${s.length - head - tail} chars] ...\n\n${s.slice(-tail)}`;
}

// ═════════════════════════════════════════════════════════════════════
//  ISkillExtractor adapter — 让 SkillConversationExtractWorker (agent 队列)
//  可以驱动 V2 extractor。Worker 持有的 ISkillExtractor 接口接受
//  conversation: ConversationMessage[] + agent_id；这里把它映射成 V2 的
//  ExtractMessage[] 形态。
// ═════════════════════════════════════════════════════════════════════

const ALLOWED_ROLES = new Set(["user", "assistant", "tool_call", "tool_result"]);

function toExtractMessages(conv: ConversationMessage[]): ExtractMessage[] {
  return conv.map((m) => ({
    role: (ALLOWED_ROLES.has(m.role) ? m.role : "user") as ExtractMessage["role"],
    content: m.content,
  }));
}

function toLegacyCandidates(items: ExtractedSkillCandidate[]): ExtractedCandidate[] {
  return items.map((c) => ({
    action: c.action as ExtractedCandidate["action"],
    name: c.name,
    skill_id: c.skill_id,
    version: c.version,
    description: c.description,
    confidence: c.confidence,
    reason: c.reason,
    file_path: c.file_path,
    file_type: c.file_type,
  }));
}

/**
 * 把 SkillExtractor 包装成 ISkillExtractor，方便 SkillConversationExtractWorker
 * (agent 队列) 直接驱动。
 *
 * 入参 input.agent_id 必须由调用方（trigger.archive 时）提供，来源是
 * SkillTaskEntry.agent_id。缺失时返回空候选并放一条 warn 日志——抽取依赖 owner
 * 校验，没有 agent_id 无法落库。
 */
export function createExtractorAdapter(
  v2: SkillExtractor,
  logger?: { warn(msg: string): void },
): ISkillExtractor {
  return {
    async extract(input) {
      const agentId = (input as { agent_id?: string }).agent_id;
      if (!agentId) {
        logger?.warn(
          `${TAG} V2 adapter: input.agent_id missing — skipping extract (returns empty candidates)`,
        );
        return { candidates: [] };
      }
      const r = await v2.extract({
        user_id: input.user_id ?? "",
        team_id: input.team_id,
        agent_id: agentId,
        task_id: input.task_id ?? input.taskId,
        // Langfuse trace 绑定字段：Worker 从 SkillTaskEntry 读到后透传到这里，
        // 缺失会让 trace sessionId=null / instanceId=unknown（按 session 筛不到）。
        session_id: input.session_id,
        space_id: input.space_id,
        messages: toExtractMessages(input.conversation),
        reason: (input as { reason?: string }).reason,
        options: (input as { options?: { max_iterations?: number } }).options,
      });
      return { candidates: toLegacyCandidates(r.candidates) };
    },
  };
}
