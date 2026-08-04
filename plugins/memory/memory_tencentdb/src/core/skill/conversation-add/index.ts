/**
 * conversation-add 模块公开入口。
 *
 * 使用方（gateway server.ts）可以直接
 *   import { wireConversationAdd, SkillConversationAddHandler, ... } from "core/skill/conversation-add/index.js";
 */

export {
  compressMessages,
  compressMessage,
  DEFAULT_COMPRESS_OPTIONS,
  type CompressibleMessage,
  type CompressOptions,
} from "./message-compressor.js";

export {
  applyOversizeStrategy,
  DEFAULT_OVERSIZE_OPTIONS,
  type OversizeMessage,
  type OversizeOptions,
  type OversizeResult,
} from "./oversize-strategy.js";

export {
  SkillBufferStorage,
  type SkillBufferStorageOptions,
  type SessionKey,
  type AgentTuple as SkillAgentTuple,
  type SessionMeta,
  type SkillTaskEntry,
  type AgentTasksDoc,
  type BufferedMessages,
} from "./buffer-storage.js";

export {
  serializeAgentTuple,
  parseAgentTuple,
  LocalSkillAgentTaskQueue,
  RedisSkillAgentTaskQueue,
  type ISkillAgentTaskQueue,
  type RedisLike as SkillAgentTaskQueueRedisLike,
  type ExtractLockHandle,
} from "./agent-task-queue.js";

export {
  SkillTriggerService,
  type SkillTriggerServiceOptions,
  type TriggerArchiveInput,
  type TriggerArchiveResult,
} from "./trigger-service.js";

export {
  SkillConversationExtractWorker,
  type SkillConversationExtractWorkerOptions,
  type SkillCandidatesSink,
} from "./extract-worker.js";

export {
  SkillConversationAddHandler,
  HandlerValidationError,
  DEFAULT_HANDLER_THRESHOLDS,
  type AddConversationInput,
  type AddConversationResult,
  type HandlerThresholds,
  type SkillConversationAddHandlerOptions,
} from "./add-handler.js";

export {
  SkillCoreSink,
  type SkillCoreSinkOptions,
  type MetadataServiceLike,
} from "./skill-core-sink.js";

export { wireConversationAdd, type WireConversationAddDeps, type WiredConversationAdd } from "./wire.js";
