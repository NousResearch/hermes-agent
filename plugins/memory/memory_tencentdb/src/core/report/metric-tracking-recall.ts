/**
 * Recall Metric Reporter — 召回阶段指标上报。
 *
 * 在召回完成后非侵入式上报以下指标到 Kafka：
 *   - recall_hit_count   : 本次召回命中的 L1 记忆条数
 *   - recall_top_score   : 召回结果中最高相似度分数（TCVDB RRF score）
 *   - recall_latency_ms  : 召回总耗时（毫秒，整数）
 *
 * 设计原则（与 MetricTrackingRunner 完全同构）：
 *   1. 召回完成后，try-catch 做上报（静默失败）
 *   2. 无论上报成功失败，不影响召回结果
 *   3. 召回失败（hasError=true）时不上报
 *   4. 无 instanceId 时不上报
 *   5. 召回 0 条时仍上报 hit_count=0 + latency，但不上报 top_score
 */

import { metricProducer } from "./kafka-metric-producer.js";

// ============================
// Strategy Encoding (numeric for ClickHouse storage)
// ============================

const STRATEGY_CODE: Record<string, number> = {
  skipped: 0,
  keyword: 1,
  embedding: 2,
  hybrid: 3,
};

// ============================
// Input Interface
// ============================

export interface RecallMetricInput {
  /** 实例 ID（Kafka key） */
  instanceId: string;
  /** 召回的 L1 记忆列表（带 score） */
  recalledL1Memories: Array<{ content: string; score: number; type: string }> | undefined;
  /** 生效的召回策略 */
  recallStrategy: string;
  /** 召回总耗时（毫秒） */
  recallLatencyMs: number;
  /** 是否召回失败 */
  hasError: boolean;
}

// ============================
// Reporter
// ============================

/**
 * 上报召回阶段指标到 Kafka。
 *
 * 静默安全：任何异常都 try-catch 吞掉，绝不向外抛。
 * 调用时机：在 performAutoRecallCore 返回后、结果传给业务前调用。
 */
export function reportRecallMetrics(input: RecallMetricInput): void {
  try {
    // Guard: 失败时不上报
    if (input.hasError) return;

    // Guard: 无 instanceId 不上报
    if (!input.instanceId) return;

    const memories = input.recalledL1Memories ?? [];
    const hitCount = memories.length;
    const latencyMs = Math.round(input.recallLatencyMs);

    // 1. 上报 recall_hit_count（含 0 条场景）
    try {
      metricProducer.send({
        metric: "recall_hit_count",
        instanceId: input.instanceId,
        value: hitCount,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 2. 上报 recall_top_score（仅有记忆时）
    if (hitCount > 0) {
      try {
        const topScore = Math.max(...memories.map((m) => m.score));
        metricProducer.send({
          metric: "recall_top_score",
          instanceId: input.instanceId,
          value: topScore,
          source: "core",
        });
      } catch {
        // 静默失败
      }
    }

    // 3. 上报 recall_latency_ms
    try {
      metricProducer.send({
        metric: "recall_latency_ms",
        instanceId: input.instanceId,
        value: latencyMs,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 4. 上报 recall_strategy（数值编码：skipped=0, keyword=1, embedding=2, hybrid=3, 未知=-1）
    try {
      const strategyCode = STRATEGY_CODE[input.recallStrategy] ?? -1;
      metricProducer.send({
        metric: "recall_strategy",
        instanceId: input.instanceId,
        value: strategyCode,
        source: "core",
      });
    } catch {
      // 静默失败
    }
  } catch {
    // 最外层 catch — 绝不向外抛
  }
}
