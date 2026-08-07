/**
 * src/core/skill/queue/index.ts
 *
 * Skill 抽取模块的**接口类型 barrel**。老 job 队列（LocalSkillTaskQueue /
 * RedisSkillTaskQueue / SkillExtractWorkerV2 / SkillExtractJob 等）已经在
 * 2026-07-17 skill_extract → 直接归档改造中删除，抽取管线现在完全走
 * `src/core/skill/conversation-add/` 下的 SkillTriggerService +
 * SkillConversationExtractWorker + SkillAgentTaskQueue。
 *
 * 本模块只保留 Worker/extractor 接口层类型（ConversationMessage /
 * ExtractedCandidate / ISkillExtractor / ExtractorLogger），给
 * conversation-add/extract-worker.ts 和 skill-extractor.ts 复用。
 */

export type {
  ConversationMessage,
  ExtractedCandidate,
  ISkillExtractor,
  ExtractorLogger,
} from "./types.js";
