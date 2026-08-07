/**
 * Offload Server — Independent type definitions.
 * This module does NOT import from src/offload/.
 */

// ─── ToolPair (ingest 写入 pending.jsonl) ────────────────────────────────────

export interface ToolPair {
  toolName: string;
  toolCallId: string;
  params: unknown;
  result: unknown;
  error?: string;
  timestamp: string;
  durationMs?: number;
}

// ─── OffloadEntry (L1 产出, entries.jsonl 每行) ──────────────────────────────

export interface OffloadEntry {
  tool_call_id: string;
  tool_call: string;
  summary: string;
  timestamp: string;
  score: number;
  node_id: string | null;
  /** Full relative path to ref file storing original tool result (e.g. "offload/{sessionId}/refs/call_214.md"), readable via tdai_read_cos */
  result_ref?: string;
}

// ─── TaskJudgment (L1.5 产出) ────────────────────────────────────────────────

export interface TaskJudgment {
  taskCompleted: boolean;
  isLongTask: boolean;
  isContinuation: boolean;
  continuationMmdFile?: string;
  newTaskLabel?: string;
}

// ─── TaskBoundary (L1.5 写入 state.boundaries) ──────────────────────────────

export interface TaskBoundary {
  targetMmd: string | null;
  timestamp: string;
}

// ─── MmdMeta (从 MMD 文件首行 %%{...}%% 解析) ──────────────────────────────

export interface MmdMeta {
  filename: string;
  taskGoal: string;
  doneCount: number;
  doingCount: number;
  todoCount: number;
  updatedTime?: string | null;
  nodeSummaries?: Array<{ nodeId: string; status: string; summary: string }>;
}

// ─── OffloadState (state.json) ───────────────────────────────────────────────

export interface OffloadState {
  activeMmdFile: string | null;
  boundaries: TaskBoundary[];
  lastL15CreatedAt: number;
}

// ─── CompactState (compact-state.json) ───────────────────────────────────────

export interface CompactState {
  confirmedOffloadIds: string[];
  deletedOffloadIds: string[];
  lastCompactedAt: string;
}

// ─── L2 Parsed Response ──────────────────────────────────────────────────────

export interface L2ParsedResponse {
  fileAction: "write" | "replace";
  mmdContent?: string;
  replaceBlocks?: Array<{
    startLine: number;
    endLine: number;
    content: string;
  }>;
  nodeMapping: Record<string, string>;
}

// ─── Configuration ───────────────────────────────────────────────────────────

export interface OffloadExecutorConfig {
  forceTriggerThreshold: number;
  pendingMaxAgeSeconds: number;

  l1Model: string;
  l1Temperature: number;
  l1MaxTokens: number;
  l1TimeoutMs: number;

  l15Model: string;
  l15Temperature: number;
  l15MaxTokens: number;
  l15TimeoutMs: number;

  l2Model: string;
  l2Temperature: number;
  l2MaxTokens: number;
  l2TimeoutMs: number;
  l2NullThreshold: number;

  mildOffloadRatio: number;
  aggressiveCompressRatio: number;
  emergencyCompressRatio: number;

  maxRetries: number;
}

export function defaultOffloadConfig(): OffloadExecutorConfig {
  return {
    forceTriggerThreshold: 4,
    pendingMaxAgeSeconds: 30,

    l1Model: "",  // uses gateway llm.model
    l1Temperature: 0.3,
    l1MaxTokens: 8000,
    l1TimeoutMs: 120_000,

    l15Model: "", // uses gateway llm.model
    l15Temperature: 0.2,
    l15MaxTokens: 3000,
    l15TimeoutMs: 120_000,

    l2Model: "",  // uses gateway llm.model
    l2Temperature: 0.4,
    l2MaxTokens: 16000,
    l2TimeoutMs: 120_000,
    l2NullThreshold: 6,

    mildOffloadRatio: 0.5,
    aggressiveCompressRatio: 0.85,
    emergencyCompressRatio: 0.95,

    maxRetries: 3,
  };
}

// ─── Default empty state ─────────────────────────────────────────────────────

export function defaultOffloadState(): OffloadState {
  return {
    activeMmdFile: null,
    boundaries: [],
    lastL15CreatedAt: 0,
  };
}

export function defaultCompactState(): CompactState {
  return {
    confirmedOffloadIds: [],
    deletedOffloadIds: [],
    lastCompactedAt: "",
  };
}
