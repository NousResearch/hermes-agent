/**
 * Knowledge entity Zod schemas + TypeScript types.
 *
 * Used by knowledge-handlers.ts for request validation.
 * Follows the same pattern as skill-schemas.ts.
 */

import { z } from "zod";

// ── Types ──

export type { KnowledgeEntity, KnowledgeType, KnowledgeListResult } from "../core/store/types.js";

// ── Schemas ──

export const knowledgeTypeSchema = z.enum(["wiki", "code-graph"]);

export const knowledgeCreateRequestSchema = z.object({
  knowledge_id: z.string().min(1),
  type: knowledgeTypeSchema,
  service_url: z.string().url(),
  name: z.string().min(1),
  summary: z.string().max(256).nullable().optional(),
  team_id: z.string().min(1),
  user_id: z.string().min(1).optional(),
  repo_url: z.string().min(1).optional(),
  branch: z.string().optional(),
});

export const knowledgeGetRequestSchema = z.object({
  knowledge_id: z.string().min(1),
  team_id: z.string().min(1).optional(),
});

export const knowledgeUpdateRequestSchema = z.object({
  knowledge_id: z.string().min(1),
  team_id: z.string().min(1).optional(),
  name: z.string().min(1).optional(),
  summary: z.string().max(256).nullable().optional(),
  service_url: z.string().url().optional(),
  repo_url: z.string().min(1).optional(),
  branch: z.string().optional(),
});

export const knowledgeBatchDeleteRequestSchema = z.object({
  knowledge_ids: z.array(z.string().min(1)).min(1).max(100),
  team_id: z.string().min(1).optional(),
});

export const knowledgeListRequestSchema = z.object({
  team_id: z.string().min(1),
  type: knowledgeTypeSchema.optional(),
  // Proxy 按 id 批量联查明细（agent 绑定解出 asset_ids 后取渲染字段）。
  knowledge_ids: z.array(z.string().min(1)).max(200).optional(),
  pagination: z.object({
    limit: z.number().int().min(1).max(1000).optional(),
    offset: z.number().int().min(0).optional(),
  }).optional(),
});
