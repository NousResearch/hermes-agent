// MVP subset of the tui_gateway JSON-RPC protocol consumed by ht-web.
// See docs/ht-web-gateway-protocol.md and ui-tui/src/gatewayTypes.ts (source of truth).

export type LiveSessionStatus = "idle" | "starting" | "waiting" | "working";
export type TranscriptRole = "user" | "assistant" | "system" | "tool";

export interface GatewaySkin {
  banner_hero?: string;
  banner_logo?: string;
  branding?: Record<string, string>;
  colors?: Record<string, string>;
  help_header?: string;
  tool_prefix?: string;
}

export interface GatewayTranscriptMessage {
  role: TranscriptRole;
  text?: string;
  name?: string;
  context?: string;
}

export interface SessionCreateResponse {
  session_id: string;
  info?: Record<string, unknown>;
}

export interface SessionResumeResponse {
  session_id: string;
  messages: GatewayTranscriptMessage[];
  info?: Record<string, unknown>;
  running?: boolean;
  status?: LiveSessionStatus;
}

export interface SessionListItem {
  id: string;
  title: string;
  preview: string;
  message_count: number;
  started_at: number;
  source?: string;
}

export interface SessionListResponse {
  sessions?: SessionListItem[];
}

export interface SessionActiveItem {
  id: string;
  status: LiveSessionStatus;
  title?: string;
  preview?: string;
  model?: string;
  message_count?: number;
  last_active?: number;
  started_at?: number;
}

export interface SessionActiveListResponse {
  sessions?: SessionActiveItem[];
}

// ── Event payloads (the subset ht-web handles) ──────────────────────────

export interface MessageDeltaPayload {
  text?: string;
  rendered?: string;
}
export interface MessageCompletePayload {
  text?: string;
  rendered?: string;
  reasoning?: string;
  usage?: Record<string, number>;
}
export interface StatusUpdatePayload {
  kind?: string;
  text?: string;
}
export interface ToolStartPayload {
  tool_id: string;
  name?: string;
  args_text?: string;
  context?: string;
}
export interface ToolProgressPayload {
  name?: string;
  preview?: string;
}
export interface ToolCompletePayload {
  tool_id: string;
  name?: string;
  result_text?: string;
  summary?: string;
  inline_diff?: string;
  error?: string;
  duration_s?: number;
}
export interface ClarifyRequestPayload {
  request_id: string;
  question: string;
  choices: string[] | null;
}
export interface ApprovalRequestPayload {
  command: string;
  description: string;
  allow_permanent?: boolean;
}
export interface ErrorPayload {
  message?: string;
}
