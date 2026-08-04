/**
 * NoopQuotaReporter — default (open-source) quota reporter.
 *
 * Behaviour:
 *   - fetchQuota() returns null → QuotaManager treats this as "unlimited",
 *     so every checkMemoryQuota / checkCreditQuota call passes.
 *   - reportUsage() is a no-op (no remote billing in open-source builds).
 *
 * Use this in standalone / self-hosted deployments where there is no
 * billing system to report into.
 */

import type { IQuotaReporter, QuotaSnapshot } from "../abstractions/index.js";

export class NoopQuotaReporter implements IQuotaReporter {
  async fetchQuota(_instanceId: string): Promise<QuotaSnapshot | null> {
    return null; // unlimited
  }

  async reportUsage(
    _instanceId: string,
    _memoryDelta: number,
    _creditDelta: number,
    _level: "L0" | "L1" | "L2" | "L3" | "Skill",
  ): Promise<void> {
    // intentionally empty — default build does not bill
  }
}
