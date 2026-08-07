/**
 * skill-permission — 权限校验纯函数
 *
 * 三种 assertion：
 *   - assertOwner: agent 是否为 head 的 owner
 *   - assertTeamMatch: row 是否属于请求的 team（不一致 → 404 不暴露存在性）
 *   - assertVersionFresh: 乐观锁检查
 *
 * 错误码与设计文档 §3.6 对齐。
 */

import type { Skill } from "./types.js";

export type SkillPermissionErrorCode =
  | "SKILL_NOT_OWNER"     // 40301
  | "SKILL_TEAM_MISMATCH" // 40302（外部行为同 NOT_FOUND，避免存在性侧信道）
  | "SKILL_NOT_FOUND"     // 40401
  | "SKILL_VERSION_STALE"; // 40901

export class SkillPermissionError extends Error {
  constructor(public readonly code: SkillPermissionErrorCode, message?: string) {
    super(message ? `${code}: ${message}` : code);
    this.name = "SkillPermissionError";
  }
}

/**
 * (teamId, agentId) 二元组必须与 headRow 匹配；否则抛 40301。
 *
 * team_id + agent_id 才唯一确定一个 agent 的 ownership——
 * 不同 team 下可能出现相同的 agent_id 值，仅校验 agent_id 不够安全。
 */
export function assertOwner(headRow: Skill, agentId: string, teamId?: string): void {
  if (teamId && headRow.team_id !== teamId) {
    throw new SkillPermissionError(
      "SKILL_NOT_OWNER",
      `team ${teamId} does not match (actual=${headRow.team_id})`,
    );
  }
  if (headRow.owner_agent_id !== agentId) {
    throw new SkillPermissionError(
      "SKILL_NOT_OWNER",
      `agent ${agentId} is not the owner (owner=${headRow.owner_agent_id})`,
    );
  }
}

/**
 * row.team_id 必须等于请求 teamId；不一致按 NOT_FOUND 处理（不暴露存在性）。
 */
export function assertTeamMatch(row: Skill | null, teamId: string): asserts row is Skill {
  if (!row || row.team_id !== teamId) {
    throw new SkillPermissionError("SKILL_NOT_FOUND");
  }
}

/**
 * 乐观锁：expected_version 必传，必须与 head.version 完全一致。
 * 不一致时抛出 SKILL_VERSION_STALE，拒绝写入，防止并发覆盖。
 */
export function assertVersionFresh(headRow: Skill, expected: number): void {
  if (expected !== headRow.version) {
    throw new SkillPermissionError(
      "SKILL_VERSION_STALE",
      `expected version ${expected}, head is ${headRow.version}`,
    );
  }
}
