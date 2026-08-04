/**
 * IStateBackend — Pipeline 状态后端抽象层
 *
 * 架构文档 §5.1 / 需求 #7.1
 *
 * Core/Worker/Timer Scanner 面向此接口编程，通过配置切换后端：
 * - LocalStateBackend  (单机，零外部依赖)
 * - RemoteStateBackend (服务化部署)
 *
 * 接口从现有 MemoryPipelineManager 中提取：
 * - Buffer    ← messageBuffers (Map<string, CapturedMessage[]>)
 * - State     ← sessionStates  (Map<string, PipelineSessionState>)
 * - Timer     ← ManagedTimer (l1Idle, l2Schedule)
 * - Queue     ← SerialQueue (l1Queue, l2Queue, l3Queue)
 * - Lock      ← l3Running / l3Pending 互斥
 * - Capture   ← notifyConversation 的 count+threshold+enqueue 原子操作
 */

// ============================
// Pipeline Session State
// ============================

/** 复用现有 checkpoint.ts 中的 PipelineSessionState 字段 */
export interface PipelineSessionState {
  conversation_count: number;
  last_extraction_time: string;
  last_extraction_updated_time: string;
  last_active_time: number;
  l2_pending_l1_count: number;
  warmup_threshold: number;
  l2_last_extraction_time: string;
}

export const DEFAULT_PIPELINE_STATE: PipelineSessionState = {
  conversation_count: 0,
  last_extraction_time: "",
  last_extraction_updated_time: "",
  last_active_time: 0,
  l2_pending_l1_count: 0,
  warmup_threshold: 0,
  l2_last_extraction_time: "",
};

// ============================
// Timer
// ============================

export interface TimerEntry {
  member: string;
  fireAtMs: number;
}

// ============================
// Task Queue
// ============================

export interface TaskPayload {
  id: string;
  type: "L1" | "L2" | "L3" | "flush" | "offload-l1" | "offload-l15" | "offload-l2";
  instanceId: string;
  sessionId: string;
  /**
   * 租户身份（可选）。v2 pipeline 用 (teamId, agentId) 决定锁粒度与
   * Redis hash tag，避免 single-instance 大 key 热点；offload 子系统不依赖。
   * 缺失时锁/key 退化到 instance 级（兼容旧调用方）。
   */
  teamId?: string;
  agentId?: string;
  priority: number; // 0=high, 1=normal, 2=low
  data?: Record<string, unknown>;
  createdAt: number;
}

// ============================
// Capture Atomic
// ============================

export interface CaptureAtomicParams {
  instanceId: string;
  sessionId: string;
  /** 同 TaskPayload.teamId / agentId — 决定 buffer + state 的 hash slot 归属。 */
  teamId?: string;
  agentId?: string;
  messageJson: string;
  threshold: number;
  fireAtMs: number;
  timerMember: string;
  taskPayload: TaskPayload;
  nowMs: number;
  /** 本次增加的对话轮数（每个 role=user 的消息算一轮）。默认 1。 */
  rounds: number;
}

export interface CaptureAtomicResult {
  triggered: boolean;
  conversationCount: number;
}

// ============================
// IStateBackend
// ============================

export interface IStateBackend {
  // ═══ Buffer ═══
  // teamId/agentId 为可选；缺失时 Redis backend hash tag 退化到 {p:inst}（旧布局）。
  // 推荐 v2 pipeline 调用方一定传入，避免单 instance 集中到一个 hash slot 形成热 key。
  appendBuffer(instanceId: string, sessionId: string, message: string, teamId?: string, agentId?: string): Promise<void>;
  drainBuffer(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<string[]>;
  getBufferLength(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<number>;

  // ═══ Session State ═══
  getSessionState(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<PipelineSessionState | null>;
  updateSessionState(instanceId: string, sessionId: string, patch: Partial<PipelineSessionState>, teamId?: string, agentId?: string): Promise<void>;
  deleteSessionState(instanceId: string, sessionId: string, teamId?: string, agentId?: string): Promise<void>;
  /**
   * 列出 instance 下所有 active session（用于 standalone 模式 persister 回放 checkpoint）。
   * Cluster 模式下 hash tag 散开后此方法只能覆盖单节点，service 模式 persister 不设置故不会调用 —— 仅作 standalone 兼容存在。
   */
  listActiveSessions(instanceId: string): Promise<string[]>;

  // ═══ Timer ═══
  setTimer(instanceId: string, member: string, fireAtMs: number): Promise<void>;
  setTimerIfEarlier(instanceId: string, member: string, fireAtMs: number): Promise<boolean>;
  removeTimer(instanceId: string, member: string): Promise<void>;
  getExpiredTimers(instanceId: string, nowMs: number): Promise<TimerEntry[]>;

  // ═══ Task Queue ═══
  enqueueTask(task: TaskPayload): Promise<void>;
  consumeTask(workerId: string, blockMs?: number): Promise<TaskPayload | null>;
  ackTask(taskId: string): Promise<void>;
  getQueueDepth(): Promise<{ high: number; low: number }>;
  /**
   * Snapshot of all tasks currently waiting in the queue (not yet consumed).
   * Used by `/v2/pipeline/status` to compute per-L-type queue stats with full
   * type/sessionId/instanceId info (queue is single-shared, but task.type
   * distinguishes L1/L2/L3 — see TaskPayload).
   *
   * Optional because remote queue implementations may opt out of expensive
   * full queue scans. Local backend (standalone) MUST implement.
   */
  listQueuedTasks?(): Promise<TaskPayload[]>;
  /** 认领超时未 ACK 的 pending 消息 (XPENDING + XCLAIM) */
  claimStaleTasks?(workerId: string, minIdleMs: number, count: number): Promise<TaskPayload[]>;

  // ═══ Lock ═══
  acquireLock(key: string, ownerId: string, ttlMs: number): Promise<boolean>;
  renewLock(key: string, ownerId: string, ttlMs: number): Promise<boolean>;
  releaseLock(key: string, ownerId: string): Promise<void>;

  // ═══ Atomic Capture ═══
  captureAtomic(params: CaptureAtomicParams): Promise<CaptureAtomicResult>;

  // ═══ Instance Lifecycle ═══
  /**
   * Purge all state associated with an instance: buffers, sessions, timers.
   * Called when an instance is destroyed.
   * @returns Counts of cleaned resources.
   */
  purgeInstance?(instanceId: string): Promise<{ sessions: number; timers: number; buffers: number }>;

  // ═══ Lifecycle ═══
  initialize?(): Promise<void>;
  destroy?(): Promise<void>;
}
