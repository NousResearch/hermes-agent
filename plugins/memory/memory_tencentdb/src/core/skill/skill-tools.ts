/**
 * SkillToolsV2 — 给 Review Agent 的工具集（绑定到 SkillCore）
 *
 * 暴露 4 个写动作 + 2 个读动作，覆盖 SkillExtractor 的需要：
 *   - skill_list      列出团队内可见 skill
 *   - skill_view      查看单个 skill 详情
 *   - skill_create    新建 skill
 *   - skill_update    全量替换 SKILL.md
 *   - skill_patch     单点串替
 *   - skill_files_write  增/改资源
 *
 * 不暴露 delete / files_remove —— 抽取流程不应能销毁团队 skill。
 * 工具错误以 JSON.stringify({error}) 返回，让 LLM 能 self-correct。
 *
 * 每次成功的写操作都 push 一条 ExtractedSkillCandidate 到 auditSink，
 * SkillExtractor 把它作为 candidates 返回给调用方。
 */

import { tool, jsonSchema } from "ai";
import { SkillCoreError, type SkillCore } from "./skill-core.js";

export type ExtractedAction =
  | "create"
  | "update"
  | "patch"
  | "files_write";

export interface ExtractedSkillCandidate {
  action: ExtractedAction;
  name: string;
  skill_id?: string;
  version?: number;
  description?: string;
}

export interface CreateSkillToolsOptions {
  core: SkillCore;
  /** 调用方身份（owner 校验依据）。 */
  user_id: string;
  team_id: string;
  agent_id: string;
  task_id?: string;
  auditSink: ExtractedSkillCandidate[];
  logger?: { info(msg: string): void; warn(msg: string): void; error(msg: string): void };
}

function jsonError(e: unknown): string {
  if (e instanceof SkillCoreError) {
    return JSON.stringify({ error: e.code, message: e.message });
  }
  return JSON.stringify({ error: "INTERNAL", message: (e as Error).message });
}

export function createSkillTools(opts: CreateSkillToolsOptions) {
  const { core, user_id, team_id, agent_id, task_id, auditSink, logger } = opts;
  // Read 路径：不带 task_id — audit 字段不参与检索。skill-core.ts:list/search
  // 内部已经再兜底 undefine 掉 task_id, 但工具层依然显式区分以让意图清晰、
  // 并防止未来 core 侧回退时又把 bug 引回来。
  // Write 路径：带 task_id 落审计列 (记 skill 首次落库时的对话上下文)。
  const readIds = { user_id, team_id, agent_id };
  const writeIds = { user_id, team_id, agent_id, task_id };

  return {
    skill_list: tool({
      description:
        "List skills available in the current team. Use this FIRST to see what already exists.",
      inputSchema: jsonSchema<{ query?: string; top_k?: number }>({
        type: "object",
        properties: {
          query: { type: "string", description: "Optional substring match on name+description" },
          top_k: { type: "number", description: "Max results (default 10)" },
        },
      }),
      execute: async ({ query, top_k }) => {
        try {
          if (query && query.trim()) {
            const hits = await core.search({ ...readIds, query, top_k });
            return JSON.stringify(hits.map((h) => ({
              skill_id: h.skill.skill_id, name: h.skill.name, description: h.skill.description,
              version: h.skill.version, score: h.score,
            })));
          }
          const r = await core.list({ ...readIds, pagination: { limit: top_k ?? 50, offset: 0 } });
          return JSON.stringify(r.items.map((s) => ({
            skill_id: s.skill_id, name: s.name, description: s.description, version: s.version,
          })));
        } catch (e) { return jsonError(e); }
      },
    }),

    skill_view: tool({
      description: "Read a skill's full SKILL.md and resource manifest.",
      inputSchema: jsonSchema<{ skill_id: string; version?: number }>({
        type: "object",
        properties: {
          skill_id: { type: "string" },
          version: { type: "number", description: "Optional historical version (default head)" },
        },
        required: ["skill_id"],
      }),
      execute: async ({ skill_id, version }) => {
        try {
          const r = await core.get({ ...readIds, skill_id, version });
          return JSON.stringify({
            skill_id: r.skill_id, version: r.version, name: r.name, description: r.description,
            content: r.content, manifest: r.manifest,
          });
        } catch (e) { return jsonError(e); }
      },
    }),

    skill_create: tool({
      description: "Create a new skill. The frontmatter `name` MUST equal the `name` parameter.",
      inputSchema: jsonSchema<{ name: string; content: string }>({
        type: "object",
        properties: {
          name: { type: "string", description: "Skill name (lowercase letters/digits/hyphen)" },
          content: { type: "string", description: "Full SKILL.md text including frontmatter" },
        },
        required: ["name", "content"],
      }),
      execute: async ({ name, content }) => {
        try {
          const r = await core.create({ ...writeIds, name, content });
          auditSink.push({ action: "create", name, skill_id: r.skill_id, version: r.version, description: r.description });
          logger?.info(`[skill-tools] created ${r.skill_id}`);
          return JSON.stringify({ ok: true, skill_id: r.skill_id, version: r.version });
        } catch (e) { return jsonError(e); }
      },
    }),

    skill_update: tool({
      description: "Replace the entire SKILL.md of an existing skill (you must own it).",
      inputSchema: jsonSchema<{ skill_id: string; content: string; expected_version: number }>({
        type: "object",
        properties: {
          skill_id: { type: "string" },
          content: { type: "string", description: "New full SKILL.md text" },
          expected_version: { type: "number", description: "Required optimistic lock — the version you just read (skill_list/skill_view). After a successful write use the returned version for the next edit." },
        },
        required: ["skill_id", "content", "expected_version"],
      }),
      execute: async ({ skill_id, content, expected_version }) => {
        try {
          const r = await core.update({ ...writeIds, skill_id, content, expected_version });
          auditSink.push({ action: "update", name: r.name, skill_id, version: r.version });
          return JSON.stringify({ ok: true, version: r.version });
        } catch (e) { return jsonError(e); }
      },
    }),

    skill_patch: tool({
      description: "Replace a unique substring in the SKILL.md. If old_string occurs >1 times you must pass replace_all.",
      inputSchema: jsonSchema<{
        skill_id: string; old_string: string; new_string: string;
        replace_all?: boolean; expected_version: number;
      }>({
        type: "object",
        properties: {
          skill_id: { type: "string" },
          old_string: { type: "string" },
          new_string: { type: "string" },
          replace_all: { type: "boolean", description: "Default false" },
          expected_version: { type: "number", description: "Required optimistic lock — the version you just read (skill_list/skill_view). After a successful write use the returned version for the next edit." },
        },
        required: ["skill_id", "old_string", "new_string", "expected_version"],
      }),
      execute: async ({ skill_id, old_string, new_string, replace_all, expected_version }) => {
        try {
          const r = await core.patch({ ...writeIds, skill_id, old_string, new_string, replace_all, expected_version });
          auditSink.push({ action: "patch", name: r.name, skill_id, version: r.version });
          return JSON.stringify({ ok: true, version: r.version });
        } catch (e) { return jsonError(e); }
      },
    }),

    skill_files_write: tool({
      description: "Write or overwrite a resource file in a skill's files/ directory.",
      inputSchema: jsonSchema<{
        skill_id: string; path: string; content: string;
        encoding?: "utf-8" | "base64"; mime_type?: string; is_executable?: boolean;
        expected_version: number;
      }>({
        type: "object",
        properties: {
          skill_id: { type: "string" },
          path: { type: "string", description: "Relative path under files/, e.g. 'scripts/run.sh'" },
          content: { type: "string" },
          encoding: { type: "string", enum: ["utf-8", "base64"] },
          mime_type: { type: "string" },
          is_executable: { type: "boolean" },
          expected_version: { type: "number", description: "Required optimistic lock — the version you just read (skill_list/skill_view). After a successful write use the returned version for the next edit." },
        },
        required: ["skill_id", "path", "content", "expected_version"],
      }),
      execute: async ({ skill_id, path, content, encoding, mime_type, is_executable, expected_version }) => {
        try {
          const r = await core.writeFiles({
            ...writeIds, skill_id, expected_version,
            files: [{ path, content, encoding: encoding ?? "utf-8", mime_type, is_executable }],
          });
          auditSink.push({ action: "files_write", name: r.name, skill_id, version: r.version });
          return JSON.stringify({ ok: true, version: r.version });
        } catch (e) { return jsonError(e); }
      },
    }),
  };
}
