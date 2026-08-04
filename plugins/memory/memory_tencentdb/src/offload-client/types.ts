/**
 * offload-client — Type definitions.
 */

// ─── Plugin Configuration ────────────────────────────────────────────────────

export interface OffloadClientConfig {
  enabled: boolean;
  /** Offload server base URL (e.g. "http://localhost:9100"). */
  serverUrl: string;
  /** Bearer token for Authorization header. */
  apiKey: string;
  /** X-TDAI-Service-Id header value. */
  serviceId: string;
  /** Agent name for sessionId construction. Default "default". */
  agentName?: string;
  /** Client-side threshold: skip compaction request when ratio < this value. Default 0.5. */
  compactionRatio: number;
  /** Ingest request timeout in ms. Default 5000. */
  ingestTimeoutMs: number;
  /** Compaction request timeout in ms. Default 30000. */
  compactionTimeoutMs: number;
}

export function defaultOffloadClientConfig(): OffloadClientConfig {
  return {
    enabled: false,
    serverUrl: "http://localhost:9100",
    apiKey: "",
    serviceId: "",
    compactionRatio: 0.5,
    ingestTimeoutMs: 5000,
    compactionTimeoutMs: 30000,
  };
}

// ─── Ingest Payload ──────────────────────────────────────────────────────────

export interface ToolPairPayload {
  toolName: string;
  toolCallId: string;
  params: unknown;
  result: unknown;
  error?: string;
  timestamp: string;
  durationMs?: number;
}

export interface RecentMessage {
  role: "user" | "assistant";
  content: string;
}

// ─── Compaction Response ─────────────────────────────────────────────────────

export interface CompactionReport {
  resolvedLevel: string;
  originalCount: number;
  compactedCount: number;
  fastPathReplaced: number;
  fastPathDeleted: number;
  mildReplacements: number;
  aggressiveDeleted: number;
  emergencyDeleted: number;
  mmdInjected: number;
}

export interface CompactionResult {
  messages: any[];
  report: CompactionReport;
}

// ─── Logger ──────────────────────────────────────────────────────────────────

export interface Logger {
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
  debug?: (...args: unknown[]) => void;
}
