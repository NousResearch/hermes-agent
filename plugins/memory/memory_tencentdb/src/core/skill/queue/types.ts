/**
 * src/core/skill/queue/types.ts
 *
 * Skill 抽取接口层类型定义。
 *
 * 2026-07-17 改造：老 job 队列（SkillExtractJob / SkillExtractResult /
 * ISkillTaskQueue / SkillQueueConfig）随着 skill_extract 收敛到
 * 「SkillTriggerService.archive → agent 队列」新链路一起删除。本文件保留
 * Worker/extractor 接口层类型（ConversationMessage / ExtractedCandidate /
 * ISkillExtractor / ExtractorLogger），供 conversation-add/extract-worker.ts
 * 与 skill-extractor.ts 复用。
 */

// ─────────────────────────────────────────────────────────────
// 队列内部用的 conversation / candidate 形状
// ─────────────────────────────────────────────────────────────

/** 队列里单条对话消息的形状（loose role 字段，兼容 user/assistant/tool_*）。 */
export interface ConversationMessage {
  role: string;
  content: string;
}

/** 抽取结果中单个 skill 候选的形状。 */
export interface ExtractedCandidate {
  action: "create" | "patch" | "edit" | "update" | "write_file" | "files_write";
  name: string;
  skill_id?: string;
  version?: number;
  description?: string;
  /** 透传字段：保留以兼容旧 result 持久化 payload。 */
  content?: string;
  old_string?: string;
  new_string?: string;
  file_path?: string;
  file_type?: "text" | "executable" | "binary";
  confidence?: number;
  reason?: string;
}

/**
 * Worker 调用的最小 extractor 接口。SkillExtractor 只需暴露这个方法。
 * 这样 queue 不依赖具体类，测试可以直接 mock。
 */
export interface ISkillExtractor {
  extract(input: {
    task_id?: string;
    taskId?: string;                // 兼容字段
    team_id: string;
    user_id?: string;
    /** owner agent id；V2 adapter 据此校验 owner 写权。 */
    agent_id?: string;
    /**
     * Langfuse 顶级 sessionId 透传。Worker 从 SkillTaskEntry.session_id 读取，
     * 一路传到 runner.run 的 telemetry metadata。缺失时 Langfuse trace 的
     * sessionId 会是 null —— 页面按 session 过滤就看不到这次 skill 抽取。
     */
    session_id?: string;
    /**
     * 实例 id（= space_id / instanceId）。透传成 runner telemetry 的 instanceId，
     * 否则 Langfuse trace 上 instanceId 会退化成 "unknown"，且 llm_call metric
     * 因 `if (params.instanceId)` 门控被跳过。
     */
    space_id?: string;
    conversation: ConversationMessage[];
    context?: { loaded_skills?: string[] };
    signal?: AbortSignal;
    /**
     * direct-trigger 场景：主 Agent 的抽取提示，透传给 extractor 注入 prompt。
     * 由 conversation-add extract-worker 从 SkillTaskEntry.reason 读取。
     */
    reason?: string;
    /** direct-trigger 场景：LLM 迭代上限，透传给 SkillExtractor 覆盖默认。 */
    options?: { max_iterations?: number };
  }): Promise<{ candidates: ExtractedCandidate[] }>;
}

/**
 * Worker 用的最小 logger 接口（兼容 console 与项目 Logger）。
 */
export interface ExtractorLogger {
  debug?: (msg: string) => void;
  info: (msg: string) => void;
  warn: (msg: string) => void;
  error: (msg: string) => void;
}

