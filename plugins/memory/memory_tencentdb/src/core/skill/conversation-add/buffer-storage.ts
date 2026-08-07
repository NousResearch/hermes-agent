/**
 * SkillBufferStorage — 封装 §4 中所有 COS 对象读写。
 *
 * 路径规则（挂在 memory 全局 PathPrefix 下，subPath 默认 "skill_buffer"）：
 *   Session 级:
 *     {subPath}/{space}/{user}/{team}/{agent}/{session}/data-current.jsonl
 *     {subPath}/{space}/{user}/{team}/{agent}/{session}/data-<ts>.jsonl
 *     {subPath}/{space}/{user}/{team}/{agent}/{session}/meta.json
 *   Agent 级:
 *     {subPath}/{space}/{user}/{team}/{agent}/_tasks.json
 *
 * 底层复用 memory 现有 StorageAdapter (Local 或 Cos)。
 *
 * 读写规则：
 *   - data-current: 明文 JSON（不做 append 语义；每次全量覆盖）
 *   - meta:         明文 JSON（session 串行，无 CAS）
 *   - archive:      明文 JSON（写入前 exists() 判定，已存在直接视为成功）
 *   - _tasks.json:  明文 JSON（读改写，由上层 SkillAgentTaskQueue 用 Redis 短锁保护）
 */

import type { StorageAdapter } from "../../storage/adapter.js";

export interface SessionKey {
  space_id: string;
  user_id: string;
  team_id: string;
  agent_id: string;
  session_id: string;
}

export interface AgentTuple {
  space_id: string;
  user_id: string;
  team_id: string;
  agent_id: string;
}

/** meta.json 结构（§4.2）。只放计数器。 */
export interface SessionMeta {
  session_id: string;
  space_id: string;
  user_id: string;
  team_id: string;
  agent_id: string;
  tool_call_count: number;
  byte_count: number;
  last_appended_at_ms?: number;
  last_archived_at_ms?: number;
}

/** _tasks.json 单个 task 条目（§4.3）。 */
export interface SkillTaskEntry {
  task_id: string;
  session_id: string;
  user_id: string;
  team_id: string;
  agent_id: string;
  space_id: string;
  task_ref_id?: string;
  archive_key: string;
  archived_at_ms: number;
  enqueued_at_ms: number;
  /**
   * direct-trigger (`/v3/skill/extract`) 独占：给 extractor prompt 注入的
   * "主 Agent 抽取提示"。conversation/add 路径不写。Worker 消费时透传给
   * `ISkillExtractor.extract({ reason })`。
   */
  reason?: string;
  /**
   * direct-trigger 独占：extractor LLM 迭代上限。conversation/add 路径不写。
   * Worker 消费时透传成 `extractor.extract({ options: { max_iterations } })`。
   */
  max_iterations?: number;
  /**
   * 累计 **permanent** 失败次数（B 类：400/422/JSON parse/schema）。
   * A 类 transient (401/403/429/5xx/网络/timeout) 不计数，避免和 B 类混淆。
   * 达到 Worker 的 `permanentMaxRetries` 阈值（默认 3）后 task 会被移到
   * `_tasks_dlq.json`（见下方 dlqKey）。
   */
  retry_count?: number;
  /**
   * 最近一次失败的 error.message（Worker 侧截断到 <=1024 字符），只用于
   * 排查；对 Worker 调度逻辑无影响。
   */
  last_error?: string;
}

/** _tasks.json 整体结构。 */
export interface AgentTasksDoc {
  team_id: string;
  agent_id: string;
  updated_at_ms: number;
  tasks: SkillTaskEntry[];
}

/**
 * `_tasks_dlq.json` 单条死信记录。
 *
 * DLQ 只落盘不做端点：人工用 `cat` / `mv` 救回来，或者 grafana 告警脚本直接
 * scan 文件。当前不做 TTL / 大小限制（每 agent 一份文件，量大时用户自己处理）。
 */
export interface SkillDeadTaskEntry extends SkillTaskEntry {
  /** DLQ 追加时的 wall clock 时间戳。 */
  dead_lettered_at_ms: number;
}

/** `_tasks_dlq.json` 整体结构。 */
export interface AgentDeadTasksDoc {
  team_id: string;
  agent_id: string;
  updated_at_ms: number;
  tasks: SkillDeadTaskEntry[];
}

/** data-current / archive 缓存内容。使用 { messages: [...] } 而不是纯 JSONL，简化读写。 */
export interface BufferedMessages {
  messages: Array<Record<string, unknown>>;
}

export interface SkillBufferStorageOptions {
  storage: StorageAdapter;
  /** COS 子路径前缀。默认 "skill_buffer"。 */
  subPath?: string;
}

const DEFAULT_SUB_PATH = "skill_buffer";

export class SkillBufferStorage {
  private readonly storage: StorageAdapter;
  private readonly subPath: string;

  constructor(opts: SkillBufferStorageOptions) {
    this.storage = opts.storage;
    this.subPath = (opts.subPath ?? DEFAULT_SUB_PATH).replace(/\/+$/, "");
  }

  // ── Path helpers ──────────────────────────────────────────────────────────

  // 路径规则对齐设计文档 §15.3：SkillBufferStorage 只负责 subPath 之下的层级
  // ({user}/{team}/{agent}/...)，space_id/instanceId 由上层 StorageAdapter 的
  // per-instance prefix 提供。带 space_id 会导致 CosStorageBackend 的 prefix
  // (`.../{instanceId}/`) 之后重复出现 `{space}/`。
  private sessionDir(sess: SessionKey): string {
    return `${this.subPath}/${sess.user_id}/${sess.team_id}/${sess.agent_id}/${sess.session_id}`;
  }

  private agentDir(agent: AgentTuple): string {
    return `${this.subPath}/${agent.user_id}/${agent.team_id}/${agent.agent_id}`;
  }

  currentKey(sess: SessionKey): string {
    return `${this.sessionDir(sess)}/data-current.jsonl`;
  }

  metaKey(sess: SessionKey): string {
    return `${this.sessionDir(sess)}/meta.json`;
  }

  archiveKey(sess: SessionKey, archivedAtMs: number): string {
    return `${this.sessionDir(sess)}/data-${archivedAtMs}.jsonl`;
  }

  tasksKey(agent: AgentTuple): string {
    return `${this.agentDir(agent)}/_tasks.json`;
  }

  dlqKey(agent: AgentTuple): string {
    return `${this.agentDir(agent)}/_tasks_dlq.json`;
  }

  // ── data-current ──────────────────────────────────────────────────────────

  async readCurrent(sess: SessionKey): Promise<BufferedMessages> {
    const raw = await this.storage.readFile(this.currentKey(sess));
    if (!raw) return { messages: [] };
    try {
      const parsed = JSON.parse(raw) as BufferedMessages;
      if (!parsed.messages) return { messages: [] };
      return { messages: parsed.messages };
    } catch {
      // 损坏 → 视为空
      return { messages: [] };
    }
  }

  async writeCurrent(sess: SessionKey, buf: BufferedMessages): Promise<void> {
    await this.storage.writeFile(this.currentKey(sess), JSON.stringify(buf));
  }

  // ── session meta.json ────────────────────────────────────────────────────

  async readMeta(sess: SessionKey): Promise<SessionMeta> {
    const raw = await this.storage.readFile(this.metaKey(sess));
    if (!raw) return this.defaultMeta(sess);
    try {
      const parsed = JSON.parse(raw) as Partial<SessionMeta>;
      return {
        ...this.defaultMeta(sess),
        ...parsed,
        // 强制关键字段一致（防止旧对象 session_id/space_id 被替换）
        session_id: sess.session_id,
        space_id: sess.space_id,
        user_id: sess.user_id,
        team_id: sess.team_id,
        agent_id: sess.agent_id,
      };
    } catch {
      return this.defaultMeta(sess);
    }
  }

  async writeMeta(sess: SessionKey, meta: SessionMeta): Promise<void> {
    await this.storage.writeFile(this.metaKey(sess), JSON.stringify(meta));
  }

  private defaultMeta(sess: SessionKey): SessionMeta {
    return {
      session_id: sess.session_id,
      space_id: sess.space_id,
      user_id: sess.user_id,
      team_id: sess.team_id,
      agent_id: sess.agent_id,
      tool_call_count: 0,
      byte_count: 0,
    };
  }

  // ── archive ────────────────────────────────────────────────────────────

  /**
   * 写归档文件；若 key 已存在直接视为成功（对齐设计 §7.4 ④）。
   *
   * 注：我们不用 If-None-Match: * 头（storage 抽象层未暴露），
   * 而是 exists() → putObject 两步。同 session 由 proxy 保证串行，
   * 且 archived_at_ms 递增（毫秒时间戳），实际不会撞。
   */
  async writeArchive(sess: SessionKey, archivedAtMs: number, buf: BufferedMessages): Promise<void> {
    const key = this.archiveKey(sess, archivedAtMs);
    if (await this.storage.exists(key)) {
      // 视为成功，跳过写入
      return;
    }
    await this.storage.writeFile(key, JSON.stringify(buf));
  }

  async readArchive(archiveKey: string): Promise<BufferedMessages | null> {
    const raw = await this.storage.readFile(archiveKey);
    if (!raw) return null;
    try {
      return JSON.parse(raw) as BufferedMessages;
    } catch {
      return null;
    }
  }

  // ── agent _tasks.json ────────────────────────────────────────────────────

  async readTasks(agent: AgentTuple): Promise<AgentTasksDoc> {
    const raw = await this.storage.readFile(this.tasksKey(agent));
    if (!raw) return this.defaultTasks(agent);
    try {
      const parsed = JSON.parse(raw) as Partial<AgentTasksDoc>;
      return {
        team_id: agent.team_id,
        agent_id: agent.agent_id,
        updated_at_ms: parsed.updated_at_ms ?? 0,
        tasks: Array.isArray(parsed.tasks) ? parsed.tasks : [],
      };
    } catch {
      return this.defaultTasks(agent);
    }
  }

  async writeTasks(agent: AgentTuple, doc: AgentTasksDoc): Promise<void> {
    await this.storage.writeFile(this.tasksKey(agent), JSON.stringify(doc));
  }

  private defaultTasks(agent: AgentTuple): AgentTasksDoc {
    return {
      team_id: agent.team_id,
      agent_id: agent.agent_id,
      updated_at_ms: 0,
      tasks: [],
    };
  }

  // ── agent _tasks_dlq.json（死信队列） ─────────────────────────────────────
  //
  // DLQ 只被 Worker 追加（且 Worker 已持 extract-lock，同一 agent 只有一个写者），
  // 因此不需要 tasks-mutex 保护——但读改写仍要求先 read 再 write，避免旧内容被截。

  async readDlq(agent: AgentTuple): Promise<AgentDeadTasksDoc> {
    const raw = await this.storage.readFile(this.dlqKey(agent));
    if (!raw) return this.defaultDlq(agent);
    try {
      const parsed = JSON.parse(raw) as Partial<AgentDeadTasksDoc>;
      return {
        team_id: agent.team_id,
        agent_id: agent.agent_id,
        updated_at_ms: parsed.updated_at_ms ?? 0,
        tasks: Array.isArray(parsed.tasks) ? parsed.tasks : [],
      };
    } catch {
      return this.defaultDlq(agent);
    }
  }

  async appendDlq(agent: AgentTuple, dead: SkillDeadTaskEntry): Promise<void> {
    const doc = await this.readDlq(agent);
    doc.tasks.push(dead);
    doc.updated_at_ms = dead.dead_lettered_at_ms;
    await this.storage.writeFile(this.dlqKey(agent), JSON.stringify(doc));
  }

  private defaultDlq(agent: AgentTuple): AgentDeadTasksDoc {
    return {
      team_id: agent.team_id,
      agent_id: agent.agent_id,
      updated_at_ms: 0,
      tasks: [],
    };
  }
}
