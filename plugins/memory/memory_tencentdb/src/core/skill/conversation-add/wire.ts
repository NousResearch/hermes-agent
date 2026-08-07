/**
 * wire.ts — one-shot wiring for a single (space/user/team/agent) or single-instance
 * conversation-add subsystem.
 *
 * 使用场景：
 *   - standalone 模式：gateway server 启动时调一次, 拿到 { handler, worker }
 *     的单例，挂到 SkillRouterDeps.resolveConversationAddHandler + Worker 常驻。
 *   - service 模式：resolveConversationAddHandler 内部按 instanceId cache
 *     其中每个 instance 各调一次 wire。
 *
 * wire 层不关心 Redis / storage 的具体来源（由 gateway 侧传入），只负责把 A1-A7
 * 和 SkillCoreSink 组装起来。
 */

import type { StorageAdapter } from "../../storage/adapter.js";
import type { ISkillExtractor, ExtractorLogger } from "../queue/types.js";

import { SkillBufferStorage } from "./buffer-storage.js";
import {
  LocalSkillAgentTaskQueue,
  RedisSkillAgentTaskQueue,
  type ISkillAgentTaskQueue,
  type RedisLike as AgentQueueRedisLike,
} from "./agent-task-queue.js";
import { SkillTriggerService } from "./trigger-service.js";
import { SkillConversationAddHandler, type HandlerThresholds } from "./add-handler.js";
import type { CompressOptions } from "./message-compressor.js";
import type { OversizeOptions } from "./oversize-strategy.js";
import {
  SkillConversationExtractWorker,
  type SkillCandidatesSink,
} from "./extract-worker.js";
import { SkillCoreSink, type MetadataServiceLike } from "./skill-core-sink.js";

export interface WireConversationAddDeps {
  storage: StorageAdapter;
  /** 生产：ioredis client；测试：LocalSkillAgentTaskQueue 直接注入到 `queue` 覆盖 */
  redis?: AgentQueueRedisLike;
  /** 测试注入的 queue；不传就基于 redis 构造 RedisSkillAgentTaskQueue */
  queue?: ISkillAgentTaskQueue;
  /** Redis key 前缀，默认 "skill" —— 与设计文档 §5 对齐 */
  redisKeyPrefix?: string;

  /**
   * sink 兜底登记 skill asset 需要的 metadata service（幂等）。
   * 不传时 sink 是 no-op —— skill 已由 extractor 的 tool-call 落库，
   * 只是前端管控页可能看不到（standalone 模式下无 onSkillCreated 钩子）。
   */
  metadataService?: MetadataServiceLike;

  /** Worker 用的 extractor，跟老 handleExtract 用同一个 */
  extractor: ISkillExtractor;

  logger: ExtractorLogger;

  /** Handler 阈值覆盖 */
  thresholds?: Partial<HandlerThresholds>;

  /** 单条 tool 消息头尾压缩规则覆盖 (对应 SkillConfig.compress)。 */
  compressOptions?: Partial<CompressOptions>;

  /** 兜底切分参数覆盖 (对应 SkillConfig.extraction 派生的 chunkMaxBytes / headKeepBytes / tailKeepBytes)。 */
  oversizeOptions?: Partial<OversizeOptions>;

  /** Worker 参数覆盖 */
  workerId?: string;
  extractLockTtlMs?: number;
  brpopBlockMs?: number;

  /** COS 子路径（默认 "skill_buffer"） */
  bufferSubPath?: string;

  /** 单元测试专用：跳过 Worker 启动（只想拿 handler / sink 时用） */
  skipWorker?: boolean;
}

export interface WiredConversationAdd {
  handler: SkillConversationAddHandler;
  worker?: SkillConversationExtractWorker;
  /**
   * 归档段服务。conversation/add 由 handler 内部持有一份直接调；
   * `/v3/skill/extract` (direct-trigger) 从这里拿来跳过 handler 的 buffer/meta 逻辑，
   * 直接归档一次 archive + 追加 `_tasks.json` + 入 agent 队列。
   */
  trigger: SkillTriggerService;
  sink: SkillCandidatesSink;
  queue: ISkillAgentTaskQueue;
  buffer: SkillBufferStorage;
  /** 结束时调 —— 关 worker */
  stop(): Promise<void>;
}

export function wireConversationAdd(deps: WireConversationAddDeps): WiredConversationAdd {
  const buffer = new SkillBufferStorage({
    storage: deps.storage,
    subPath: deps.bufferSubPath,
  });

  let queue: ISkillAgentTaskQueue;
  if (deps.queue) {
    queue = deps.queue;
  } else if (deps.redis) {
    queue = new RedisSkillAgentTaskQueue({
      client: deps.redis,
      keyPrefix: deps.redisKeyPrefix ?? "skill",
    });
  } else {
    // 无 Redis 的降级：仅 standalone 单机场景可用
    queue = new LocalSkillAgentTaskQueue();
    deps.logger.warn(
      "[skill-conversation-add] no redis nor queue injected — falling back to in-memory queue (single-node only)",
    );
  }

  // [obs] SkillTriggerService / SkillConversationAddHandler 内部走 obsLogger 底座，
  // 不再需要注入 logger —— obsLogger 自带 FileLogger + 后端 + try/catch 降级。
  const trigger = new SkillTriggerService({ buffer, queue });
  const handler = new SkillConversationAddHandler({
    buffer,
    trigger,
    thresholds: deps.thresholds,
    compressOptions: deps.compressOptions,
    oversizeOptions: deps.oversizeOptions,
  });

  const sink = new SkillCoreSink({
    metadata: deps.metadataService,
    logger: deps.logger,
  });

  let worker: SkillConversationExtractWorker | undefined;
  if (!deps.skipWorker) {
    worker = new SkillConversationExtractWorker({
      workerId: deps.workerId ?? `skill-conv-worker-${process.pid}`,
      buffer,
      queue,
      extractor: deps.extractor,
      sink,
      logger: deps.logger,
      extractLockTtlMs: deps.extractLockTtlMs,
      brpopBlockMs: deps.brpopBlockMs,
    });
    worker.start();
  }

  return {
    handler,
    worker,
    trigger,
    sink,
    queue,
    buffer,
    stop: async () => {
      if (worker) await worker.stop();
    },
  };
}
