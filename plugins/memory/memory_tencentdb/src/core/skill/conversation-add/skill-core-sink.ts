/**
 * SkillCoreSink — 把 SkillConversationExtractWorker 抽出的 candidates 兜底
 * 登记 skill asset。
 *
 * ⚠️ 语义澄清（跟老 `/v3/skill/extract` 同步分支的兜底路径完全对齐）：
 *   SkillExtractor 内部走的是 tool-calling review agent —— agent 通过
 *   `<skill_tools>` 里的 create tool 直接调 `SkillCore.create` 把 skill 落库了，
 *   candidate 只是"事后回执"（带上 skill_id / name / action="create"）。
 *
 *   所以 sink **不应该** 再调一遍 SkillCore.create，否则会：
 *     1) 同名/同 team 冲突 → SKILL_NAME_DUPLICATE 抛错
 *     2) candidate 里 content 通常也没有（extractor 用 tool-call 参数直接落，
 *        result payload 只保 skill_id / name 摘要）
 *
 *   sink 只做兜底 asset 登记 —— standalone 模式下 SkillVersioning 没挂
 *   onSkillCreated 钩子（避免 core 耦合 metadata），要靠这里补登记 asset；
 *   service 模式下 buildSkillCore 已挂钩子，这里调 ensureSkillAsset 幂等，
 *   属于双保险。
 */

import type { ExtractedCandidate, ExtractorLogger } from "../queue/types.js";
import type { SkillCandidatesSink } from "./extract-worker.js";

/** 与 gateway 侧 MetadataService 使用的形状对齐（只用到 ensureSkillAsset）。 */
export interface MetadataServiceLike {
  ensureSkillAsset(input: {
    skill_id: string;
    team_id: string;
    agent_id: string;
    name: string;
  }): Promise<unknown>;
}

export interface SkillCoreSinkOptions {
  /** 可选 —— 有 metadata service 时兜底登记 asset。缺失时 sink 是 no-op。 */
  metadata?: MetadataServiceLike;
  logger: ExtractorLogger;
}

/**
 * SkillCoreSink 只做 asset 兜底登记, 不再 create skill。
 * skill 本身已由 SkillExtractor 的 tool-call review agent 通过 SkillCore.create 落库。
 */
export class SkillCoreSink implements SkillCandidatesSink {
  constructor(private readonly opts: SkillCoreSinkOptions) {}

  async applyCandidates(input: {
    task: {
      team_id: string;
      user_id: string;
      agent_id: string;
      task_ref_id?: string;
      session_id: string;
    };
    candidates: ExtractedCandidate[];
    workerId: string;
  }): Promise<void> {
    const { metadata, logger } = this.opts;
    const { task, candidates, workerId } = input;
    if (!candidates.length) return;
    if (!metadata) return; // 没有 metadata → sink 是 no-op（asset 登记只能靠钩子）

    for (const c of candidates) {
      if (c.action !== "create") {
        // 目前只 asset 化 create；patch/update 不需要重新登记 asset（skill_id 不变）
        continue;
      }
      const skillId = c.skill_id;
      const name = c.name;
      if (!skillId || !name) {
        logger.warn(
          `[skill-core-sink] worker=${workerId} candidate missing skill_id/name — skip asset register`,
        );
        continue;
      }
      try {
        await metadata.ensureSkillAsset({
          skill_id: skillId,
          team_id: task.team_id,
          agent_id: task.agent_id,
          name,
        });
      } catch (err) {
        // asset 登记失败不影响主流程 —— skill 已经在 skills 表里，
        // 前端管控页可能显示不到，运维再补
        logger.warn(
          `[skill-core-sink] worker=${workerId} ensureSkillAsset failed skill_id=${skillId}: ` +
            (err instanceof Error ? err.message : String(err)),
        );
      }
    }
  }
}
