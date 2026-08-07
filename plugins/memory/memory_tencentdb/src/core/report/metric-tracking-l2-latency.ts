/**
 * L2 Scene Extraction Metric Reporter — L2 场景提取指标上报。
 *
 * 在 L2 场景提取完成后非侵入式上报以下指标到 Kafka：
 *   - l2_extraction_latency_ms  : L2 场景提取端到端总耗时（毫秒）
 *   - l2_llm_duration_ms       : L2 LLM 调用耗时（毫秒）
 *   - l2_scene_count_before    : 提取前场景数量
 *   - l2_scene_count_after     : 提取后场景数量
 *   - l2_scenes_created        : 本次新增场景数
 *   - l2_scenes_updated        : 本次更新场景数
 *   - l2_scenes_deleted        : 本次删除场景数
 *
 * Trace 关联说明：
 *   所有指标通过 metricProducer.send() 自动注入当前 active span 的 traceId。
 *   L2 执行在 TracedTaskExecutor.executeL2() → withSpan("core.l2.extraction") 中，
 *   其子 span 包含 Vercel AI SDK 自动产生的 ai.generateText / gen_ai.chat span，
 *   携带 gen_ai.request.model 属性。通过 traceId 可反查到：
 *     - 用了什么模型
 *     - LLM 请求/响应详情
 *     - 整个 L2 提取链路的每个环节耗时
 *
 * 设计原则：
 *   1. 提取完成后，try-catch 做上报（静默失败）
 *   2. 无论上报成功失败，不影响提取结果
 *   3. 提取失败（hasError=true）时不上报
 *   4. 无 instanceId 时不上报
 *   5. llmDurationMs 为 null 表示无 LLM 调用计时，不上报 l2_llm_duration_ms
 */

import { metricProducer } from "./kafka-metric-producer.js";

// ============================
// Input Interface
// ============================

export interface L2LatencyMetricInput {
  /** 实例 ID（Kafka key） */
  instanceId: string;
  /** L2 场景提取端到端总耗时（毫秒） */
  extractionLatencyMs: number;
  /** LLM 调用耗时（毫秒）。null 表示无 LLM 调用计时 */
  llmDurationMs: number | null;
  /** 提取前场景数量 */
  sceneCountBefore: number;
  /** 提取后场景数量 */
  sceneCountAfter: number;
  /** 本次新增场景数 */
  scenesCreated: number;
  /** 本次更新场景数 */
  scenesUpdated: number;
  /** 本次删除场景数 */
  scenesDeleted: number;
  /** 是否提取失败 */
  hasError: boolean;
}

// ============================
// Reporter
// ============================

/**
 * 上报 L2 场景提取指标到 Kafka。
 *
 * 静默安全：任何异常都 try-catch 吞掉，绝不向外抛。
 */
export function reportL2LatencyMetrics(input: L2LatencyMetricInput): void {
  try {
    // Guard: 失败时不上报
    if (input.hasError) return;

    // Guard: 无 instanceId 不上报
    if (!input.instanceId) return;

    // 1. 上报 l2_extraction_latency_ms
    try {
      metricProducer.send({
        metric: "l2_extraction_latency_ms",
        instanceId: input.instanceId,
        value: Math.round(input.extractionLatencyMs),
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 2. 上报 l2_llm_duration_ms（仅有 LLM 计时时）
    if (input.llmDurationMs !== null) {
      try {
        metricProducer.send({
          metric: "l2_llm_duration_ms",
          instanceId: input.instanceId,
          value: Math.round(input.llmDurationMs),
          source: "core",
        });
      } catch {
        // 静默失败
      }
    }

    // 3. 上报 l2_scene_count_before
    try {
      metricProducer.send({
        metric: "l2_scene_count_before",
        instanceId: input.instanceId,
        value: input.sceneCountBefore,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 4. 上报 l2_scene_count_after
    try {
      metricProducer.send({
        metric: "l2_scene_count_after",
        instanceId: input.instanceId,
        value: input.sceneCountAfter,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 5. 上报 l2_scenes_created
    try {
      metricProducer.send({
        metric: "l2_scenes_created",
        instanceId: input.instanceId,
        value: input.scenesCreated,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 6. 上报 l2_scenes_updated
    try {
      metricProducer.send({
        metric: "l2_scenes_updated",
        instanceId: input.instanceId,
        value: input.scenesUpdated,
        source: "core",
      });
    } catch {
      // 静默失败
    }

    // 7. 上报 l2_scenes_deleted
    try {
      metricProducer.send({
        metric: "l2_scenes_deleted",
        instanceId: input.instanceId,
        value: input.scenesDeleted,
        source: "core",
      });
    } catch {
      // 静默失败
    }
  } catch {
    // 最外层 catch — 绝不向外抛
  }
}
