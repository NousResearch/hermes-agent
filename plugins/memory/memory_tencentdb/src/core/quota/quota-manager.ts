/**
 * QuotaManager — 配额管理器
 *
 * 职责:
 * 1. 缓存并检查 MemoryLimit / CreditLimit 是否超限
 * 2. 通过注入的 IQuotaReporter 上报用量变化
 * 3. 本地缓存 Usage 避免每次请求都调远程
 *
 * 配额数据来源由依赖注入的 IQuotaReporter 决定：
 *   - standalone: NoopQuotaReporter (fetchQuota 返回 null → 视为无限额)
 *   - service:    由部署环境注入远程配额 reporter
 */

import type { Logger } from "../logger.js";
import type { IQuotaReporter } from "../abstractions/index.js";

export interface QuotaConfig {
  memoryLimit: number;   // 记忆总条数上限 (default: 10000)
  creditLimit: number;   // Credit 限额 (default: 1000)
  memoryUsage: number;   // 当前已用记忆条数
  creditUsage: number;   // 当前已用 Credit
}

export interface QuotaCheckResult {
  allowed: boolean;
  reason?: "memory_limit_exceeded" | "credit_limit_exceeded";
  current?: number;
  limit?: number;
}

export interface QuotaManagerOptions {
  /** Pre-constructed quota reporter for the current deployment. */
  reporter: IQuotaReporter;
  /** 配额缓存 TTL (毫秒), 默认 60s */
  cacheTtlMs?: number;
  /** 默认 MemoryLimit (上游未返回时使用) */
  defaultMemoryLimit?: number;
  /** 默认 CreditLimit (上游未返回时使用) */
  defaultCreditLimit?: number;
  logger: Logger;
}

const TAG = "[quota-manager]";

export class QuotaManager {
  private reporter: IQuotaReporter;
  private logger: Logger;
  private cacheTtlMs: number;
  private defaultMemoryLimit: number;
  private defaultCreditLimit: number;

  // Per-instance 缓存
  private cache = new Map<string, { config: QuotaConfig; expiresAt: number }>();

  constructor(opts: QuotaManagerOptions) {
    this.reporter = opts.reporter;
    this.logger = opts.logger;
    this.cacheTtlMs = opts.cacheTtlMs ?? 60_000;
    this.defaultMemoryLimit = opts.defaultMemoryLimit ?? 10_000;
    this.defaultCreditLimit = opts.defaultCreditLimit ?? 1_000;
  }

  /**
   * 获取实例配额配置 (带缓存)
   *
   * 当 reporter.fetchQuota() 返回 null (开源/无配额模式), 视为无限额 ——
   * 返回 memoryUsage=0/creditUsage=0 + defaultLimit, 这样所有 check 都会通过。
   */
  async getQuota(instanceId: string): Promise<QuotaConfig> {
    const now = Date.now();
    const cached = this.cache.get(instanceId);
    if (cached && now < cached.expiresAt) {
      return cached.config;
    }

    try {
      const snapshot = await this.reporter.fetchQuota(instanceId);

      if (snapshot === null) {
        // 无配额模式 (Noop reporter): 返回默认 limit + 零 usage, 永远不会超限
        const config: QuotaConfig = {
          memoryLimit: this.defaultMemoryLimit,
          creditLimit: this.defaultCreditLimit,
          memoryUsage: 0,
          creditUsage: 0,
        };
        this.cache.set(instanceId, { config, expiresAt: now + this.cacheTtlMs });
        return config;
      }

      const config: QuotaConfig = {
        memoryLimit: snapshot.memoryLimit,
        creditLimit: snapshot.creditLimit,
        memoryUsage: snapshot.memoryUsage,
        creditUsage: snapshot.creditUsage,
      };
      this.cache.set(instanceId, { config, expiresAt: now + this.cacheTtlMs });
      return config;
    } catch (err) {
      this.logger.warn(`${TAG} Failed to fetch quota for ${instanceId}: ${err instanceof Error ? err.message : String(err)}`);
      return this.getDefaultOrCached(instanceId);
    }
  }

  /**
   * 检查是否允许写入记忆 (MemoryUsage < MemoryLimit)
   */
  async checkMemoryQuota(instanceId: string, delta: number = 1): Promise<QuotaCheckResult> {
    const quota = await this.getQuota(instanceId);
    // limit < 0 表示不限额（兼容控制面用 -1 表达 unlimited）。
    if (quota.memoryLimit >= 0 && quota.memoryUsage + delta > quota.memoryLimit) {
      return {
        allowed: false,
        reason: "memory_limit_exceeded",
        current: quota.memoryUsage,
        limit: quota.memoryLimit,
      };
    }
    return { allowed: true };
  }

  /**
   * 检查是否允许使用 LLM (CreditUsage < CreditLimit)
   */
  async checkCreditQuota(instanceId: string): Promise<QuotaCheckResult> {
    const quota = await this.getQuota(instanceId);
    // limit < 0 表示不限额（兼容控制面用 -1 表达 unlimited）。
    if (quota.creditLimit >= 0 && quota.creditUsage >= quota.creditLimit) {
      return {
        allowed: false,
        reason: "credit_limit_exceeded",
        current: quota.creditUsage,
        limit: quota.creditLimit,
      };
    }
    return { allowed: true };
  }

  /**
   * 上报用量变化 (通过注入的 reporter)
   * @param memoryDelta 记忆条数变化 (正=新增, 负=删除)
   * @param creditDelta Credit 消耗变化 (正=消耗)
   * @param level 记忆层级 ("L0" | "L1" | "L2" | "L3")
   */
  async reportUsage(instanceId: string, memoryDelta: number, creditDelta: number, level: "L0" | "L1" | "L2" | "L3" | "Skill" = "L0"): Promise<void> {
    if (memoryDelta === 0 && creditDelta === 0) return;

    // Reporter 内部保证不抛错; 这里仍然 try/catch 以防接口契约被违反
    try {
      await this.reporter.reportUsage(instanceId, memoryDelta, creditDelta, level);

      // 同步更新本地缓存 (无论 reporter 是 noop 还是真实上报, 本地都需要追踪)
      const cached = this.cache.get(instanceId);
      if (cached) {
        cached.config.memoryUsage += memoryDelta;
        cached.config.creditUsage += creditDelta;
      }

      this.logger.debug?.(`${TAG} Usage reported: instance=${instanceId}, memDelta=${memoryDelta}, creditDelta=${creditDelta}`);
    } catch (err) {
      // Defensive: reporter 不应该抛错, 但万一抛了不能影响业务
      this.logger.error(`${TAG} reportUsage unexpected error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }

  /**
   * 快捷方法: 上报记忆新增
   */
  async reportMemoryAdded(instanceId: string, count: number, level: "L0" | "L1" | "L2" | "L3" = "L0"): Promise<void> {
    return this.reportUsage(instanceId, count, 0, level);
  }

  /**
   * 快捷方法: 上报记忆删除
   */
  async reportMemoryDeleted(instanceId: string, count: number, level: "L0" | "L1" | "L2" | "L3" = "L0"): Promise<void> {
    return this.reportUsage(instanceId, -count, 0, level);
  }

  /**
   * 快捷方法: 上报 Credit 消耗
   */
  async reportCreditUsed(instanceId: string, credits: number, level: "L0" | "L1" | "L2" | "L3" = "L1"): Promise<void> {
    return this.reportUsage(instanceId, 0, credits, level);
  }

  /**
   * 快捷方法: 上报 Skill VDB 新增 (新建 skill / 新增版本)
   */
  async reportSkillAdded(instanceId: string, count: number = 1): Promise<void> {
    return this.reportUsage(instanceId, count, 0, "Skill");
  }

  /**
   * 快捷方法: 上报 Skill VDB 删除 (软删除 / TTL 清理)
   */
  async reportSkillDeleted(instanceId: string, count: number = 1): Promise<void> {
    return this.reportUsage(instanceId, -count, 0, "Skill");
  }

  /** 清除缓存 (测试用) */
  clearCache(): void {
    this.cache.clear();
  }

  private getDefaultOrCached(instanceId: string): QuotaConfig {
    const cached = this.cache.get(instanceId);
    if (cached) return cached.config; // 用过期的旧缓存
    return {
      memoryLimit: this.defaultMemoryLimit,
      creditLimit: this.defaultCreditLimit,
      memoryUsage: 0,
      creditUsage: 0,
    };
  }
}
