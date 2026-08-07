/**
 * Kafka Metric Producer — core 组件（门面层）
 *
 * 异步发送监控指标消息，由 IMetricBackend 后端处理。
 * 内部环境：Kafka → memory-monitor 消费聚合后上报 Barad + ClickHouse。
 * 开源环境：Noop 或 Console 输出。
 *
 * 设计要点：
 *   - 异步发送，不阻塞业务请求
 *   - 发送失败静默忽略（不重试、不阻塞）
 *   - 不修改任何业务代码，纯可观测性组件
 *
 * 使用方式：
 *   import { metricProducer } from "../core/report/kafka-metric-producer.js";
 *   metricProducer.send({ metric: "l1_extraction_credit_rate", instanceId: "mem-abc", value: 150 });
 *
 * 公开 API 签名保持不变，调用方无需修改。
 */

import { getObservabilityBackend } from "./factory.js";
import type { MetricMessage, MetricBackendConfig } from "./types.js";

// 重导出类型（保持向后兼容）
export type { MetricMessage } from "./types.js";
export type KafkaMetricConfig = MetricBackendConfig;

// ============================
// CRC32 Partition 计算（保留导出，供私有模块使用）
// ============================

import CRC32 from "crc-32";

/**
 * 使用 CRC32 IEEE 计算 Partition 编号。
 * 与 Go 端 `crc32.ChecksumIEEE([]byte(instanceId)) % totalPartitions` 结果一致。
 *
 * 注意：crc-32 npm 包返回有符号 32 位整数，需要转为无符号。
 */
export function calculatePartition(instanceId: string, totalPartitions: number): number {
  if (totalPartitions <= 0) return 0;
  // crc-32 返回有符号 int32，转为无符号 uint32
  const checksum = CRC32.str(instanceId) >>> 0;
  return checksum % totalPartitions;
}

// ============================
// Metric Producer 门面
// ============================

/**
 * Metric Producer 门面。
 *
 * 保持与原 KafkaMetricProducer 相同的公开 API：
 * - send(msg) — 发送一条指标消息
 * - initialize(config) — 初始化后端
 * - destroy() — 优雅关闭
 *
 * 内部委托给 IMetricBackend（通过全局单例获取）。
 */
class MetricProducerFacade {
  /**
   * 异步发送一条监控消息。
   * 如果后端未初始化或已关闭，静默忽略。
   *
   * 自动注入 traceId：从当前 OTel active span 中提取 traceId，
   * 注入到 MetricMessage 中（仅当调用方未手动传入时）。
   * 注入失败时静默降级为空字符串，不影响 metric 发送。
   */
  send(msg: MetricMessage): void {
    try {
      // 自动注入 traceId（调用方手动传入时不覆盖）
      if (msg.traceId === undefined) {
        try {
          const ctx = getObservabilityBackend().tracePropagation.serializeTraceContext();
          msg = { ...msg, traceId: (ctx as Record<string, unknown>)._traceId as string ?? "" };
        } catch {
          msg = { ...msg, traceId: "" };
        }
      }
      getObservabilityBackend().metric.send(msg);
    } catch {
      // 静默失败
    }
  }

  /**
   * 初始化 Metric 后端。
   * 实际初始化由 initObservabilityBackend() 统一完成，
   * 此方法保留用于向后兼容。
   */
  async initialize(config: MetricBackendConfig): Promise<void> {
    try {
      await getObservabilityBackend().metric.initialize(config);
    } catch {
      // 静默失败
    }
  }

  /**
   * 优雅关闭。
   */
  async destroy(): Promise<void> {
    try {
      await getObservabilityBackend().metric.destroy();
    } catch {
      // 静默失败
    }
  }
}

/** 全局单例 */
export const metricProducer = new MetricProducerFacade();