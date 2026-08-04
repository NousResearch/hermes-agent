/**
 * TimerScanner — Sharded Timer Scanner (Scheme D + Mode 1)
 *
 * Architecture:
 * - Timers are stored in 16 sharded global ZSETs: `{prefix}:timers:shard_{0..15}`
 * - Member format: `{instanceId}\x00{sessionId}:{timerType}`, score = fireAtMs
 * - ALL pods run the scanner (no leader election needed)
 * - Each pod scans all 16 shards using Lua atomic claim (ZRANGEBYSCORE + ZREM)
 * - Lua atomicity guarantees no duplicate consumption across pods
 *
 * Performance:
 * - Fixed number of shard-claim calls per scan interval
 * - O(1) per scan regardless of instance count
 * - Each shard keeps bounded member counts to avoid hot keys
 *
 * Usage:
 *   Embedded: const scanner = new TimerScanner(backend, config); scanner.start();
 */

import type { IStateBackend, TaskPayload, TimerEntry } from "../core/state/types.js";
import { parsePipelineTimerMember } from "../core/state/timer-member.js";

interface Logger {
  debug?: (message: string) => void;
  info: (message: string) => void;
  warn: (message: string) => void;
  error: (message: string) => void;
}

export interface TimerScannerConfig {
  /** 扫描间隔 ms (default: 2000) */
  scanIntervalMs?: number;
  /** 每个 shard 每次最多取出的 timer 数 (default: 1000) */
  claimBatchSize?: number;
  /** 节点 ID (用于日志标识) */
  nodeId?: string;
  /** Legacy: 实例列表（Scheme D 下不再需要，保留兼容） */
  instances?: string[] | (() => Promise<string[]>);
  /** Legacy: leader 相关配置（Scheme D 下忽略） */
  leaderLockKey?: string;
  leaderLockTtlMs?: number;
  leaderRenewIntervalMs?: number;
}

const TAG = "[timer-scanner]";

// ============================
// TimerScanner
// ============================

export class TimerScanner {
  private backend: IStateBackend;
  private config: {
    scanIntervalMs: number;
    claimBatchSize: number;
    nodeId: string;
  };
  private logger: Logger;

  private scanTimer: ReturnType<typeof setInterval> | null = null;
  private destroyed = false;

  // Metrics
  private metrics = {
    scansCompleted: 0,
    tasksEnqueued: 0,
    scanErrors: 0,
    lastScanMs: 0,
    lastScanAt: 0,
    isLeader: true, // Always true in Scheme D (all pods are "leaders")
  };

  constructor(backend: IStateBackend, config: TimerScannerConfig, logger?: Logger) {
    this.backend = backend;
    this.logger = logger ?? { info: console.log, warn: console.warn, error: console.error };
    this.config = {
      scanIntervalMs: config.scanIntervalMs ?? 2000,
      claimBatchSize: config.claimBatchSize ?? 1000,
      nodeId: config.nodeId ?? `scanner-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`,
    };
  }

  // ============================
  // Lifecycle
  // ============================

  async start(): Promise<void> {
    if (this.destroyed) return;
    this.logger.info(`${TAG} Starting (nodeId=${this.config.nodeId}, interval=${this.config.scanIntervalMs}ms, shards=${this.getShardCount()})`);
    this.startScanLoop();
    this.logger.info(`${TAG} Started (leaderless mode, all pods scan)`);
  }

  async stop(): Promise<void> {
    if (this.destroyed) return;
    this.destroyed = true;
    this.stopScanLoop();
    this.logger.info(`${TAG} Stopped (scans=${this.metrics.scansCompleted}, enqueued=${this.metrics.tasksEnqueued})`);
  }

  getMetrics() {
    return { ...this.metrics, nodeId: this.config.nodeId };
  }

  // ============================
  // Scan Loop
  // ============================

  private startScanLoop(): void {
    if (this.scanTimer) return;
    this.scanTimer = setInterval(() => this.scan(), this.config.scanIntervalMs);
  }

  private stopScanLoop(): void {
    if (this.scanTimer) {
      clearInterval(this.scanTimer);
      this.scanTimer = null;
    }
  }

  private getShardCount(): number {
    // Use backend-specific shard count if available
    const rb = this.backend as any;
    if (typeof rb.timerShardCount === "number") return rb.timerShardCount;
    return 16; // default
  }

  private getShardKey(shard: number): string {
    const rb = this.backend as any;
    if (typeof rb.getTimerShardKeyByIndex === "function") return rb.getTimerShardKeyByIndex(shard);
    return `tdai_memory:timers:shard_${shard}`;
  }

  private async scan(): Promise<void> {
    if (this.destroyed) return;

    const startMs = Date.now();
    try {
      const shardCount = this.getShardCount();
      const now = Date.now();
      let totalEnqueued = 0;

      for (let shard = 0; shard < shardCount; shard++) {
        const rb = this.backend as any;
        let expired: TimerEntry[];

        if (typeof rb.claimExpiredFromShard === "function") {
          // Scheme D: atomic claim from shard
          expired = await rb.claimExpiredFromShard(this.getShardKey(shard), now, this.config.claimBatchSize);
        } else {
          // Fallback for LocalStateBackend: use legacy getExpiredTimers
          break; // LocalStateBackend handles timers internally via setTimeout
        }

        for (const entry of expired) {
          const { instanceId, sessionId, taskType, priority, timerType, teamId, agentId } = this.parseShardMember(entry.member);

          const task: TaskPayload = {
            id: `${taskType}-${instanceId.slice(-8)}-${sessionId.slice(-8)}-${now}`,
            type: taskType,
            instanceId,
            sessionId,
            teamId,
            agentId,
            priority,
            createdAt: now,
            data: { triggeredBy: "timer_scanner", timerMember: `${sessionId}:${timerType}`, instanceId, teamId, agentId },
          };

          await this.backend.enqueueTask(task);
          totalEnqueued++;

          this.logger?.debug?.(
            `${TAG} [${instanceId}] Timer expired: ${sessionId}:${taskType} → enqueued ${taskType} task`,
          );
        }
      }

      this.metrics.scansCompleted++;
      this.metrics.tasksEnqueued += totalEnqueued;
      this.metrics.lastScanMs = Date.now() - startMs;
      this.metrics.lastScanAt = Date.now();

      if (totalEnqueued > 0) {
        this.logger.info(`${TAG} Scan complete: enqueued ${totalEnqueued} task(s) in ${this.metrics.lastScanMs}ms`);
      }
    } catch (err) {
      this.metrics.scanErrors++;
      this.logger.error(`${TAG} Scan error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * Parse shard member format: "{instanceId}\x00{sessionId}:{timerType}"
   * Example: "mem-j4wjesud\x00sess_001:L1_idle" → { instanceId: "mem-j4wjesud", sessionId: "sess_001", taskType: "L1" }
   */
  private parseShardMember(member: string): { instanceId: string; sessionId: string; taskType: TaskPayload["type"]; priority: number; timerType: string; teamId?: string; agentId?: string } {
    const sep = member.indexOf("\x00");
    let instanceId: string;
    let rest: string;

    if (sep >= 0) {
      instanceId = member.slice(0, sep);
      rest = member.slice(sep + 1);
    } else {
      // Fallback: try colon-separated (legacy format "instanceId:sessionId:type")
      const firstColon = member.indexOf(":");
      instanceId = member.slice(0, firstColon);
      rest = member.slice(firstColon + 1);
    }

    // rest = timer member after instanceId separator
    // New unified format: "offload-{type}:{embeddedInstanceId}:{sessionId}[:{extra}]" (prefix-based)
    // Legacy format: "sessionId:L1_idle" or "sessionId:L2_schedule"

    // Check for offload prefix format first
    if (rest.startsWith("offload-l15:")) {
      // Skip embedded instanceId: "offload-l15:{instanceId}:{sessionId}"
      const afterPrefix = rest.slice("offload-l15:".length);
      const colonIdx = afterPrefix.indexOf(":");
      const sessionId = colonIdx > 0 ? afterPrefix.slice(colonIdx + 1) : afterPrefix;
      return { instanceId, sessionId, taskType: "offload-l15", priority: 0, timerType: rest };
    }
    if (rest.startsWith("offload-l2:")) {
      // Skip embedded instanceId: "offload-l2:{instanceId}:{sessionId}[:{mmdFile}]"
      const afterPrefix = rest.slice("offload-l2:".length);
      const colonIdx = afterPrefix.indexOf(":");
      let sessionId = colonIdx > 0 ? afterPrefix.slice(colonIdx + 1) : afterPrefix;
      // Strip trailing ":{mmdFile}" from sessionId
      if (sessionId.endsWith(".mmd")) {
        const lastColon = sessionId.lastIndexOf(":");
        if (lastColon > 0) sessionId = sessionId.slice(0, lastColon);
      }
      return { instanceId, sessionId, taskType: "offload-l2", priority: 1, timerType: rest };
    }
    if (rest.startsWith("offload-l1:")) {
      // Skip embedded instanceId: "offload-l1:{instanceId}:{sessionId}"
      const afterPrefix = rest.slice("offload-l1:".length);
      const colonIdx = afterPrefix.indexOf(":");
      const sessionId = colonIdx > 0 ? afterPrefix.slice(colonIdx + 1) : afterPrefix;
      return { instanceId, sessionId, taskType: "offload-l1", priority: 0, timerType: rest };
    }

    const parsed = parsePipelineTimerMember(rest);
    return { instanceId, ...parsed };
  }
}

// ============================
// Standalone entry point
// ============================

export async function startTimerScanner(): Promise<TimerScanner> {
  const { createStateBackend } = await import("../core/state/index.js");

  const backend = await createStateBackend({
    type: (process.env.STATE_BACKEND as "redis" | "local") || "redis",
    redis: {
      host: process.env.REDIS_HOST || "127.0.0.1",
      port: parseInt(process.env.REDIS_PORT || "6379", 10),
      password: process.env.REDIS_PASSWORD || undefined,
      keyPrefix: process.env.REDIS_KEY_PREFIX || "tdai_memory",
    },
  });

  const scanner = new TimerScanner(backend, {
    scanIntervalMs: parseInt(process.env.SCANNER_INTERVAL_MS || "2000", 10),
    nodeId: process.env.SCANNER_NODE_ID,
  });

  const shutdown = async () => {
    await scanner.stop();
    await backend.destroy?.();
    process.exit(0);
  };
  process.on("SIGTERM", shutdown);
  process.on("SIGINT", shutdown);

  await scanner.start();
  return scanner;
}
