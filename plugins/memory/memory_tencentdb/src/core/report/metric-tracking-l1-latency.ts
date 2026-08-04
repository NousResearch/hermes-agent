/**
 * L1 Latency Metric Reporter — L1 提取阶段延迟指标上报。
 *
 * 在 L1 提取完成后非侵入式上报以下指标到 Kafka：
 *   - l1_extraction_latency_ms : L1 提取端到端总耗时（毫秒）
 *   - l1_dedup_latency_ms     : 去重阶段耗时（毫秒）
 *
 * 设计原则（与 metric-tracking-recall 完全同构）：
 *   1. 提取完成后，try-catch 做上报（静默失败）
 *   2. 无论上报成功失败，不影响提取结果
 *   3. 提取失败（hasError=true）时不上报
 *   4. 无 instanceId 时不上报
 *   5. dedupLatencyMs 为 null 表示未走去重路径，不上报 dedup 指标
 */

import { metricProducer } from "./kafka-metric-producer.js";

// ============================
// Input Interface
// ============================

export interface L1LatencyMetricInput {
  /** 实例 ID（Kafka key） */
  instanceId: string;
  /** L1 提取端到端总耗时（毫秒） */
  extractionLatencyMs: number;
  /** 去重阶段耗时（毫秒）。null 表示未走去重路径 */
  dedupLatencyMs: number | null;
  /** 是否提取失败 */
  hasError: boolean;
}

// ============================
// Reporter
// ============================

/**
 * 上报 L1 提取阶段延迟指标到 Kafka。
 *
 * 静默安全：任何异常都 try-catch 吞掉，绝不向外抛。
 */
export function reportL1LatencyMetrics(input: L1LatencyMetricInput): void {
  try {
    // Guard: 失败时不上报
    if (input.hasError) return;

    // Guard: 无 instanceId 不上报
    if (!input.instanceId) return;

    // 1. 上报 l1_extraction_latency_ms
    try {
      metricProducer.send({
        metric: "l1_extraction_latency_ms",
        instanceId: input.instanceId,
        value: Math.round(input.extractionLatencyMs),
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 2. 上报 l1_dedup_latency_ms（仅走了去重路径时）
    if (input.dedupLatencyMs !== null) {
      try {
        metricProducer.send({
          metric: "l1_dedup_latency_ms",
          instanceId: input.instanceId,
          value: Math.round(input.dedupLatencyMs),
          source: "core",
        });
      } catch {
        // 静默失败
      }
    }
  } catch {
    // 最外层 catch — 绝不向外抛
  }
}
