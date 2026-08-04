/**
 * L3 Persona Generation Metric Reporter — L3 画像生成指标上报。
 *
 * 在 L3 画像生成完成后非侵入式上报以下指标到 Kafka：
 *   - l3_generation_latency_ms : L3 画像生成端到端总耗时（毫秒）
 *   - persona_length_before    : 更新前画像字符数（初始生成时为 0）
 *   - persona_length_after     : 更新后画像字符数
 *
 * Trace 关联说明：
 *   所有指标通过 metricProducer.send() 自动注入当前 active span 的 traceId。
 *   L3 执行在 TracedTaskExecutor.executeL3() → withSpan("core.l3.generation") 中，
 *   其子 span 包含 Vercel AI SDK 自动产生的 ai.generateText / gen_ai.chat span，
 *   携带 gen_ai.request.model 属性。通过 traceId 可反查到：
 *     - 用了什么模型（gen_ai.request.model）
 *     - LLM prompt 和 response 详情
 *     - 画像生成的完整调用链路
 *     - 该次 LLM 调用的耗时
 *
 * 设计原则：
 *   1. 生成完成后，try-catch 做上报（静默失败）
 *   2. 无论上报成功失败，不影响生成结果
 *   3. 生成失败（hasError=true）时不上报
 *   4. 无 instanceId 时不上报
 */

import { metricProducer } from "./kafka-metric-producer.js";

// ============================
// Input Interface
// ============================

export interface L3LatencyMetricInput {
  /** 实例 ID（Kafka key） */
  instanceId: string;
  /** L3 画像生成端到端总耗时（毫秒） */
  generationLatencyMs: number;
  /** 更新前画像字符数（初始生成时传 0） */
  personaLengthBefore: number;
  /** 更新后画像字符数 */
  personaLengthAfter: number;
  /** 更新前画像纯文本（用于计算 drift ratio，可选） */
  personaTextBefore?: string;
  /** 更新后画像纯文本（用于计算 drift ratio，可选） */
  personaTextAfter?: string;
  /** 是否生成失败 */
  hasError: boolean;
}

// ============================
// Reporter
// ============================

/**
 * 上报 L3 画像生成指标到 Kafka。
 *
 * 静默安全：任何异常都 try-catch 吞掉，绝不向外抛。
 */
export function reportL3LatencyMetrics(input: L3LatencyMetricInput): void {
  try {
    // Guard: 失败时不上报
    if (input.hasError) return;

    // Guard: 无 instanceId 不上报
    if (!input.instanceId) return;

    // 1. 上报 l3_generation_latency_ms
    try {
      metricProducer.send({
        metric: "l3_generation_latency_ms",
        instanceId: input.instanceId,
        value: Math.round(input.generationLatencyMs),
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 2. 上报 persona_length_before
    try {
      metricProducer.send({
        metric: "persona_length_before",
        instanceId: input.instanceId,
        value: input.personaLengthBefore,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 3. 上报 persona_length_after
    try {
      metricProducer.send({
        metric: "persona_length_after",
        instanceId: input.instanceId,
        value: input.personaLengthAfter,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 4. 上报 persona_drift_ratio（行级 diff）
    //    计算方式：将新旧画像按行切分，统计新增行数+删除行数 / 总行数
    //    值域 [0, 1]：0 = 完全没变，1 = 完全重写
    //    仅在有 before/after 文本且非首次生成时计算
    try {
      if (input.personaTextBefore != null && input.personaTextAfter != null && input.personaTextBefore.length > 0) {
        const driftRatio = computeLineDriftRatio(input.personaTextBefore, input.personaTextAfter);
        metricProducer.send({
          metric: "persona_drift_ratio",
          instanceId: input.instanceId,
          value: driftRatio,
          source: "core",
        });
      }
    } catch {
      // 静默失败
    }
  } catch {
    // 最外层 catch — 绝不向外抛
  }
}

// ============================
// Helpers
// ============================

/**
 * 计算行级 drift ratio（方案 C）。
 *
 * 将新旧画像按行切分（忽略空行），计算：
 *   drift = (新增行数 + 删除行数) / max(旧行数 + 新行数, 1)
 *
 * 值域 [0, 1]：
 *   0   = 内容完全一致
 *   ~0.1 = 微调（改了几行）
 *   ~0.5 = 大幅修改
 *   1   = 完全重写（没有一行相同）
 *
 * 性能：O(n) 时间 + O(n) 空间，画像通常 50-200 行，耗时 < 0.1ms。
 */
export function computeLineDriftRatio(before: string, after: string): number {
  const oldLines = before.split("\n").filter((l) => l.trim().length > 0);
  const newLines = after.split("\n").filter((l) => l.trim().length > 0);

  const oldSet = new Set(oldLines);
  const newSet = new Set(newLines);

  const added = newLines.filter((l) => !oldSet.has(l)).length;
  const removed = oldLines.filter((l) => !newSet.has(l)).length;

  const total = oldLines.length + newLines.length;
  if (total === 0) return 0;

  // 归一化到 [0, 1]
  return Math.min((added + removed) / total, 1);
}
