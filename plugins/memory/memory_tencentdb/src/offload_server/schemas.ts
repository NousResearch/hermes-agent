/**
 * Offload Server — Request validation schemas (Zod).
 */
import { z } from "zod";

/** Safe session ID: alphanumeric, underscore, hyphen, dot, colon allowed. No slashes or path traversal. Max 500 chars. */
const safeSessionId = z.string().min(1).max(500, {
  message: "sessionId must not exceed 500 characters",
}).regex(/^[a-zA-Z0-9_.\-:]+$/, {
  message: "Must only contain alphanumeric, underscore, hyphen, dot, or colon characters",
});

const ToolPairSchema = z.object({
  tool_name: z.string(),
  tool_call_id: z.string(),
  params: z.unknown(),
  result: z.unknown(),
  error: z.string().optional(),
  timestamp: z.string(),
  duration_ms: z.number().optional(),
});

/** Recent message item: user/assistant text only (no tool_call/tool_result). */
const RecentMessageSchema = z.object({
  role: z.enum(["user", "assistant"]),
  content: z.string(),
});

export const IngestRequestSchema = z
  .object({
    session_id: safeSessionId,
    tool_pairs: z.array(ToolPairSchema).default([]),
    /** Current user prompt that triggers L1.5 task judgment. Must be non-empty (whitespace-only is rejected). */
    prompt: z.string().trim().min(1, { message: "prompt must not be empty or whitespace-only" }).optional(),
    /** Recent history messages (user/assistant only, no tool calls). */
    recent_messages: z.array(RecentMessageSchema).optional(),
  })
  .refine(
    (data) => data.tool_pairs.length > 0 || (data.prompt && data.prompt.length > 0),
    { message: "Either tool_pairs must be non-empty or prompt must be provided" },
  );

export type IngestRequest = z.infer<typeof IngestRequestSchema>;

/**
 * Each message must have non-empty `role` and `content` fields.
 * Intentionally lenient: any non-empty string role is accepted to support
 * OpenAI / Anthropic / OpenClaw-wrapped formats without over-specifying.
 */
const CompactionMessageSchema = z
  .record(z.string(), z.unknown())
  .refine(
    (msg) => typeof msg["role"] === "string" && (msg["role"] as string).length > 0,
    { message: "Each message must have a non-empty 'role' field" },
  );

export const CompactionRequestSchema = z.object({
  session_id: safeSessionId,
  messages: z.array(CompactionMessageSchema),
  ratio: z.number().min(0).max(2),
});

export type CompactionRequest = z.infer<typeof CompactionRequestSchema>;

/** Extended compaction schema with token metadata for L3 compression. */
export const CompactionRequestSchemaV2 = z.object({
  session_id: safeSessionId,
  messages: z.array(CompactionMessageSchema),
  ratio: z.number().min(0).max(2),
  context_window: z.number().int().min(1),
  total_tokens: z.number().int().min(0),
  message_tokens: z.array(z.number()).optional(),
});

export type CompactionRequestV2 = z.infer<typeof CompactionRequestSchemaV2>;

export const MmdQuerySchema = z.object({
  session_id: safeSessionId,
  limit: z.number().int().min(1).optional(),
});
