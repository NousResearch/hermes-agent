/**
 * Core type definitions for the context offload plugin.
 * Ported from context-offload-plugin with updated runtime defaults.
 */

import type { Logger } from "../core/types.js";

// ============================
// Data types
// ============================

/** A single offloaded tool call/result summary stored in offload.jsonl */
export interface OffloadEntry {
  /** ISO timestamp inherited from the original tool result */
  timestamp: string;
  /** Mermaid node ID assigned by L2, null until L2 runs */
  node_id: string | null;
  /** Short description of the tool call command */
  tool_call: string;
  /** LLM-generated summary of the tool result */
  summary: string;
  /** Relative path to the MD file containing the full tool result */
  result_ref: string;
  /** The original tool call ID from the provider */
  tool_call_id: string;
  /** Session key this entry belongs to */
  session_key?: string;
  /** Replaceability score (0-10). Higher = summary can better replace original. Assigned by L1 LLM. */
  score?: number;
}

/** A buffered tool call + result pair waiting to be processed by L1 */
export interface ToolPair {
  toolName: string;
  toolCallId: string;
  params: Record<string, unknown> | string;
  result: unknown;
  error?: string;
  timestamp: string;
  durationMs?: number;
}

/** Persistent plugin state saved to state.json */
export interface PluginState {
  /** Path to the currently active MMD file (relative to mmds/) */
  activeMmdFile: string | null;
  /** Identifier/label for the active MMD */
  activeMmdId: string | null;
  /** Counter for auto-incrementing MMD filenames */
  mmdCounter: number;
  /** Last session key the plugin was active in */
  lastSessionKey: string | null;
  /** Last tool_call_id that was successfully offloaded into compact context (L3 cursor) */
  lastOffloadedToolCallId: string | null;
  /** ISO timestamp of the last successful L2 trigger */
  lastL2TriggerTime: string | null;
}

/** Metadata block embedded in MMD files */
export interface MmdMetadata {
  taskGoal: string;
  createdTime: string;
  updatedTime: string;
}

/** A node in the Mermaid flowchart */
export interface MmdNode {
  id: string;
  label: string;
  status: "done" | "doing" | "todo";
  summary: string;
  timestamp: string;
}

// ============================
// LLM types
// ============================

/** Configuration for the LLM client */
export interface LlmConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

/** Result from L1.5 task judgment */
export interface TaskJudgment {
  /** Whether the current task is completed */
  taskCompleted: boolean;
  /** Whether the new task is a continuation of a recent task */
  isContinuation: boolean;
  /** If continuation, which MMD file to reactivate */
  continuationMmdFile?: string;
  /** Short label for new task (used in MMD filename) */
  newTaskLabel?: string;
  /** Whether this is a long task (vs. casual chat) */
  isLongTask: boolean;
}

/** L1.5 boundary marker: divides entries into task-attributed segments.
 *  Each boundary defines the ownership of entries from startIndex onward
 *  until the next boundary's startIndex. */
export interface L15Boundary {
  /** Entry counter value when L1.5 judgment started.
   *  Entries at this index and beyond belong to this boundary's result. */
  startIndex: number;
  /** L1.5 judgment result for this segment */
  result: "long" | "short" | "pending";
  /** If result="long", the target MMD file for L2 to construct into */
  targetMmd: string | null;
}

/** Result from an LLM call */
export interface LlmResponse {
  content: string;
  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;
  };
}

/** OpenClaw config model provider shape (minimal) */
export interface ModelProvider {
  baseUrl?: string;
  apiKey?: string;
  models?: Record<string, unknown>;
}

// ============================
// Plugin configuration
// ============================

/**
 * Plugin configuration, read from openclaw.json -> plugins.entries config.
 * All fields are optional; defaults are used when not specified.
 */
export interface PluginConfig {
  /** Explicit LLM model for offload tasks, format: "provider/model-id" (e.g. "dashscope/kimi-k2.5") */
  model?: string;
  /** LLM temperature for offload tasks. Default: 0.2 */
  temperature?: number;
  /** Force-trigger L1 when pending tool pairs >= this threshold. Default: 4 */
  forceTriggerThreshold?: number;
  /** Custom data directory path (absolute). Default: ~/.openclaw/context-offload */
  dataDir?: string;
  /** Default context window size when not found in model config. Default: 200000 */
  defaultContextWindow?: number;
  /** Max tool pairs to process per L1 batch. Default: 20 */
  maxPairsPerBatch?: number;
  /** Trigger L2 when offload.jsonl has >= this many node_id=null entries. Default: 4 */
  l2NullThreshold?: number;
  /** Trigger L2 if it hasn't run for this many seconds. Default: 300 (5 minutes) */
  l2TimeoutSeconds?: number;
  /**
   * If L2 leaves entries in `node_id="wait"` (e.g. parse/mapping failure),
   * those entries will be retried after waiting for at least this many seconds.
   * Default: 120
   */
  l2WaitRetrySeconds?: number;
  /**
   * If true (default), time-based L2 only runs when at least one `node_id=null` entry has
   * `timestamp` strictly after `lastL2TriggerTime` (i.e. new offload rows since last L2).
   * Does not affect condition A (null count threshold). Set false for legacy timeout retry of stale nulls.
   */
  l2TimeTriggerRequiresNewOffload?: boolean;
  /** Mild offload: replace non-current-task tool results when context >= this ratio. Default: 0.5 */
  mildOffloadRatio?: number;
  /** Mild offload scan range: scan the last N% of messages (0.7 = last 70%). Default: 0.7 */
  mildOffloadScanRatio?: number;
  /** Mild offload phase-1: replace top N% highest-score (most replaceable) entries first. Default: 0.4 */
  mildScoreTopRatio?: number;
  /** Mild offload: only trigger when current task messages occupy >= this ratio of total tokens. Default: 0.8 */
  mildCurrentTaskRatio?: number;
  /** Aggressive compress: delete tail messages when context >= this ratio. Default: 0.85 */
  aggressiveCompressRatio?: number;
  /**
   * Aggressive compress: target fraction of **message** tokens to remove from the **oldest**
   * messages each round (0.4 ≈ oldest 40% of total per-message token sum). Default: 0.4
   */
  aggressiveDeleteRatio?: number;
  /** Emergency trigger: when tokens >= contextWindow * emergencyCompressRatio, fire emergency. Default: 0.95 */
  emergencyCompressRatio?: number;
  /** Emergency target: delete until tokens <= contextWindow * emergencyTargetRatio. Default: 0.6 */
  emergencyTargetRatio?: number;
  /** Max ratio of total tokens that injected MMDs may occupy. Default: 0.2 */
  mmdMaxTokenRatio?: number;
  /**
   * L3 token counting: `tiktoken` uses js-tiktoken (exact BPE for chosen encoding);
   * `heuristic` uses 中文/1.7 + 其余/4. Default: tiktoken.
   */
  l3TokenCountMode?: "tiktoken" | "heuristic";
  /**
   * tiktoken encoding when `l3TokenCountMode` is `tiktoken`.
   * Typical: `o200k_base` (GPT-4o/o-series), `cl100k_base` (GPT-4/3.5). Default: o200k_base.
   */
  l3TiktokenEncoding?:
    | "gpt2"
    | "r50k_base"
    | "p50k_base"
    | "p50k_edit"
    | "cl100k_base"
    | "o200k_base";
  /**
   * Default ratio of context window assumed to be system overhead (system prompt +
   * tool schemas). Used when no cached overhead is available from llm_input hook.
   * Default: 0.12 (12%).
   */
  defaultSystemOverheadRatio?: number;
}

// ============================
// Logger interface
// ============================

/** Logger interface used by offload plugin components */
export type PluginLogger = Logger;

// ============================
// Plugin defaults
// ============================

/** Defaults for all configurable values (sourced from runtime .js) */
export const PLUGIN_DEFAULTS = {
  temperature: 0.2,
  forceTriggerThreshold: 4,
  defaultContextWindow: 200_000,
  maxPairsPerBatch: 20,
  l2NullThreshold: 4,
  l2TimeoutSeconds: 300,
  /** If L2 leaves entries in node_id="wait", retry after this many seconds */
  l2WaitRetrySeconds: 120,
  /** When true, time-based L2 only fires if some node_id=null row is newer than last L2 */
  l2TimeTriggerRequiresNewOffload: true,
  mildOffloadRatio: 0.5,
  mildOffloadScanRatio: 0.7,
  mildScoreTopRatio: 0.4,
  mildCurrentTaskRatio: 0.8,
  aggressiveCompressRatio: 0.85,
  aggressiveDeleteRatio: 0.4,
  /** Emergency trigger: when tokens >= contextWindow * 0.95, fire emergency */
  emergencyCompressRatio: 0.95,
  /** Emergency target: delete until tokens <= contextWindow * 0.6 */
  emergencyTargetRatio: 0.6,
  mmdMaxTokenRatio: 0.2,
  l3TokenCountMode: "tiktoken" as const,
  l3TiktokenEncoding: "cl100k_base" as const,
  defaultSystemOverheadRatio: 0.12,
} as const;
