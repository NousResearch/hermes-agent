/**
 * MetricTrackingRunner / MetricTrackingRunnerFactory — LLMRunner 装饰器，
 * 在 LLM 调用完成后非侵入式上报 credit 消耗到 Kafka。
 *
 * 设计原则（与 MetricTrackingStore 完全同构）：
 *   1. 先执行原方法，拿到结果
 *   2. 原方法成功后，try-catch 做上报（静默失败）
 *   3. 无论上报成功失败，都返回原方法的结果
 *   4. 原方法抛异常时直接 re-throw，不执行任何上报
 *   5. 不改变 LLMRunner / LLMRunnerFactory 接口签名
 *
 * taskId → 指标名映射：
 *   - "l1-extraction"        → "l1_extraction_credit_rate"
 *   - "l1-conflict-detection" → "l1_dedup_credit_rate"
 *   - "scene-extract-*"      → "l2_extraction_credit_rate"
 *   - "persona-generation"   → "l3_generation_credit_rate"
 *
 * Token Usage 获取策略：
 *   - 优先从 inner runner 的 lastUsage side-channel 读取精确值（区分 input/output）
 *   - 如果不可用（OpenClaw 路径），基于字符长度估算
 *
 * Credit 计算（Producer 侧完成，Consumer 侧只做 ÷窗口周期得速率）：
 *   公式：Credit = (input_tokens/10000 × INPUT_RATE + output_tokens/10000 × OUTPUT_RATE) × multiplier
 *   1 Credit = 10000 个标准 Input Tokens（以 M2.7 为锚点）
 *   基础费率：
 *     - INPUT_RATE  = 1.0 Credit / 10k tokens
 *     - CACHE_RATE  = 0.2 Credit / 10k tokens（一期暂不区分 cache，全按 input 计）
 *     - OUTPUT_RATE = 4.0 Credit / 10k tokens
 *   模型系数：M2.7 = 1.0，旗舰型 = 15.0，极速型 = 0.8（一期默认 1.0）
 *   降级策略：无法区分 input/output 时，全按 input 费率（1.0）保守估算
 */

import type {
  LLMRunner,
  LLMRunParams,
  LLMRunnerFactory,
  LLMRunnerCreateOptions,
} from "../types.js";
import { metricProducer } from "./kafka-metric-producer.js";

// ============================
// taskId → 指标名映射
// ============================

/** LLM Runner 的 token usage 信息（side-channel） */
export interface LLMUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/** 带 lastUsage side-channel 的 LLMRunner（可选扩展） */
export interface LLMRunnerWithUsage extends LLMRunner {
  lastUsage?: LLMUsage;
}

/**
 * 根据 taskId 映射到 Kafka 指标名。
 * 返回 undefined 表示该 taskId 不需要上报 credit。
 */
export function taskIdToMetricName(taskId: string): string | undefined {
  if (taskId === "l1-extraction") return "l1_extraction_credit_rate";
  if (taskId === "l1-conflict-detection") return "l1_dedup_credit_rate";
  if (taskId.startsWith("scene-extract")) return "l2_extraction_credit_rate";
  if (taskId === "persona-generation") return "l3_generation_credit_rate";
  return undefined;
}

/**
 * 根据 taskId 映射到评测 token 指标前缀。
 * 返回 undefined 表示该 taskId 不需要按环节上报 token。
 *
 * 产出指标名举例：
 *   l1_extraction_input_tokens / l1_extraction_output_tokens
 *   l1_dedup_input_tokens / l1_dedup_output_tokens
 *   l2_extraction_input_tokens / l2_extraction_output_tokens
 *   l3_generation_input_tokens / l3_generation_output_tokens
 */
export function taskIdToTokenMetricPrefix(taskId: string): string | undefined {
  if (taskId === "l1-extraction") return "l1_extraction";
  if (taskId === "l1-conflict-detection") return "l1_dedup";
  if (taskId.startsWith("scene-extract")) return "l2_extraction";
  if (taskId === "persona-generation") return "l3_generation";
  return undefined;
}

// ============================
// Credit 计算常量与函数
// ============================

/** 1 Credit 对应的 token 基数（10000 tokens = 1 Credit） */
export const TOKENS_PER_CREDIT = 10000;
/** 基础费率：标准输入 1.0 Credit / 10k tokens（M2.7 锚点） */
export const INPUT_RATE = 1.0;
/** 基础费率：缓存输入 0.2 Credit / 10k tokens（一期暂不区分 cache，全按 input 计） */
export const CACHE_RATE = 0.2;
/** 基础费率：模型输出 4.0 Credit / 10k tokens */
export const OUTPUT_RATE = 4.0;
/** 默认模型系数（M2.7 标准型） */
export const DEFAULT_MULTIPLIER = 1.0;

/**
 * 将 Credit 值四舍五入到 5 位小数。
 * 所有上报统一使用此函数，保证精度一致。
 */
export function roundCredit(value: number): number {
  return Math.round(value * 100000) / 100000;
}

/**
 * 根据 taskId 映射到记忆层级（用于用量上报）。
 * 返回 undefined 表示该 taskId 不需要上报。
 */
export function taskIdToLevel(taskId: string): "L1" | "L2" | "L3" | undefined {
  if (taskId === "l1-extraction" || taskId === "l1-conflict-detection") return "L1";
  if (taskId.startsWith("scene-extract")) return "L2";
  if (taskId === "persona-generation") return "L3";
  return undefined;
}

/** onCreditConsumed 回调参数 */
export interface CreditConsumedEvent {
  instanceId: string;
  credit: number;
  level: "L1" | "L2" | "L3";
  taskId: string;
}

/** onCreditConsumed 回调类型 */
export type OnCreditConsumed = (event: CreditConsumedEvent) => void;

/**
 * 根据精确的 input/output token 数计算 Credit 值。
 * 公式：Credit = (input/10000 × INPUT_RATE + output/10000 × OUTPUT_RATE) × multiplier
 *
 * 1 Credit = 10000 个标准 Input Tokens。
 * 一期简化：不区分 cache tokens，全按 input 费率计算。
 * 后续 LLM SDK 支持返回 cache hit 数后再精细化。
 */
export function computeCredit(
  inputTokens: number,
  outputTokens: number,
  multiplier: number = DEFAULT_MULTIPLIER,
): number {
  return ((inputTokens / TOKENS_PER_CREDIT) * INPUT_RATE + (outputTokens / TOKENS_PER_CREDIT) * OUTPUT_RATE) * multiplier;
}

/**
 * 基于字符长度粗略估算 token 数量，然后计算 Credit。
 * 英文约 4 字符/token，中文约 2 字符/token，取折中值 3 字符/token。
 *
 * 降级策略：无法区分 input/output 时，全按 input 费率（1.0）保守估算。
 * 公式：Credit = estimatedTotalTokens / 10000 × INPUT_RATE × multiplier
 */
export function estimateCreditFromChars(
  inputCharLength: number,
  outputCharLength: number,
  multiplier: number = DEFAULT_MULTIPLIER,
): number {
  const estimatedTokens = Math.ceil((inputCharLength + outputCharLength) / 3);
  return (estimatedTokens / TOKENS_PER_CREDIT) * INPUT_RATE * multiplier;
}

// ============================
// MetricTrackingRunner（装饰器）
// ============================

/**
 * 包装 LLMRunner，在 run() 完成后异步上报 credit 消耗到 Kafka。
 *
 * 上报的 value 是 **Credit 值**（已完成 Token → Credit 转换），
 * Consumer 侧只需 ÷ 窗口周期得速率，不需要再做 Token → Credit 换算。
 *
 * 安全保证：
 *   - 上报失败静默忽略，绝不影响 run() 的返回值
 *   - 原方法抛异常时不上报，异常正常传播
 *   - 不改变 run() 的签名和返回值
 */
export class MetricTrackingRunner implements LLMRunner {
  private readonly inner: LLMRunner;
  private readonly getInstanceId: () => string | undefined;
  private readonly multiplier: number;
  private readonly onCreditConsumed?: OnCreditConsumed;

  /** Accumulated credit consumed across all run() calls on this runner instance. */
  accumulatedCredit = 0;

  constructor(
    inner: LLMRunner,
    getInstanceId: () => string | undefined,
    multiplier: number = DEFAULT_MULTIPLIER,
    onCreditConsumed?: OnCreditConsumed,
  ) {
    this.inner = inner;
    this.getInstanceId = getInstanceId;
    this.multiplier = multiplier;
    this.onCreditConsumed = onCreditConsumed;
  }

  async run(params: LLMRunParams): Promise<string> {
    // 0. 注入 instanceId（如果调用方没传，从 getInstanceId 回调获取）
    const enrichedParams = params.instanceId
      ? params
      : { ...params, instanceId: this.getInstanceId() };

    // 1. 先执行原方法，拿到结果（异常直接 re-throw）
    const text = await this.inner.run(enrichedParams);

    // 2. 原方法成功后，try-catch 做上报（静默失败）
    try {
      const metricName = taskIdToMetricName(params.taskId);
      if (metricName) {
        const instanceId = enrichedParams.instanceId ?? this.getInstanceId();
        if (instanceId) {
          // 优先从 inner runner 的 lastUsage side-channel 读取精确 token 数
          const innerWithUsage = this.inner as LLMRunnerWithUsage;
          let creditValue: number;
          let inputTokens: number;
          let outputTokens: number;

          if (innerWithUsage.lastUsage && innerWithUsage.lastUsage.totalTokens > 0) {
            inputTokens = innerWithUsage.lastUsage.promptTokens;
            outputTokens = innerWithUsage.lastUsage.completionTokens;
            creditValue = computeCredit(inputTokens, outputTokens, this.multiplier);
          } else {
            // 无精确 token 数时，基于字符长度估算
            const inputChars = (params.prompt?.length ?? 0) + (params.systemPrompt?.length ?? 0);
            const outputChars = text.length;
            inputTokens = Math.ceil(inputChars / 3);
            outputTokens = Math.ceil(outputChars / 3);
            creditValue = estimateCreditFromChars(inputChars, outputChars, this.multiplier);
          }

          // 统一 round 到 5 位小数（所有上报使用相同数据）
          const roundedCredit = roundCredit(creditValue);

          // Accumulate credit for caller to read
          this.accumulatedCredit += roundedCredit;

          if (roundedCredit > 0) {
            // 上报聚合指标（5 位小数，静默失败）
            try {
              metricProducer.send({
                metric: metricName,
                instanceId,
                value: roundedCredit,
                source: "core",
              });
            } catch {
              // 指标发送失败静默忽略
            }

            // 上报用量回调（同样 5 位小数，静默失败）
            if (this.onCreditConsumed) {
              const level = taskIdToLevel(params.taskId);
              if (level) {
                try {
                  this.onCreditConsumed({
                    instanceId,
                    credit: roundedCredit,
                    level,
                    taskId: params.taskId,
                  });
                } catch {
                  // 静默失败，绝不影响业务
                }
              }
            }
          }

          // 上报原始 Token 指标（用于聚合侧计算 TPM）
          // 只有 > 0 的指标才上报，静默失败
          try {
            if (inputTokens > 0) {
              metricProducer.send({
                metric: "llm_input_tokens",
                instanceId,
                value: inputTokens,
                source: "core",
              });
            }
            if (outputTokens > 0) {
              metricProducer.send({
                metric: "llm_output_tokens",
                instanceId,
                value: outputTokens,
                source: "core",
              });
            }
          } catch {
            // Token 指标上报失败静默忽略，绝不影响业务
          }

          // 上报按环节区分的 Token 指标（评测用，带 traceId）
          try {
            const tokenPrefix = taskIdToTokenMetricPrefix(params.taskId);
            if (tokenPrefix && inputTokens > 0) {
              metricProducer.send({
                metric: `${tokenPrefix}_input_tokens`,
                instanceId,
                value: inputTokens,
                source: "core",
              });
            }
            if (tokenPrefix && outputTokens > 0) {
              metricProducer.send({
                metric: `${tokenPrefix}_output_tokens`,
                instanceId,
                value: outputTokens,
                source: "core",
              });
            }
          } catch {
            // 静默失败
          }
        }
      }
    } catch {
      // 静默失败，绝不影响业务
    }

    // 3. 无论上报成功失败，都返回原方法的结果
    return text;
  }
}

// ============================
// MetricTrackingRunnerFactory（装饰器）
// ============================

/**
 * 包装 LLMRunnerFactory，创建出的 Runner 自带 credit 上报能力。
 *
 * 注入点：在 tdai-core.ts 的 wirePipelineRunners() 中包装 factory。
 * 这是唯一的"改动"——属于可观测性代码的注入点，不是业务逻辑的修改。
 *
 * @param multiplier 模型系数（默认 1.0 = M2.7 标准型）。
 *   后续可从配置中读取，支持多模型动态切换。
 */
export class MetricTrackingRunnerFactory implements LLMRunnerFactory {
  private readonly inner: LLMRunnerFactory;
  private readonly getInstanceId: () => string | undefined;
  private readonly multiplier: number;
  private readonly onCreditConsumed?: OnCreditConsumed;

  constructor(
    inner: LLMRunnerFactory,
    getInstanceId: () => string | undefined,
    multiplier: number = DEFAULT_MULTIPLIER,
    onCreditConsumed?: OnCreditConsumed,
  ) {
    this.inner = inner;
    this.getInstanceId = getInstanceId;
    this.multiplier = multiplier;
    this.onCreditConsumed = onCreditConsumed;
  }

  createRunner(opts?: LLMRunnerCreateOptions): LLMRunner {
    const innerRunner = this.inner.createRunner(opts);
    return new MetricTrackingRunner(innerRunner, this.getInstanceId, this.multiplier, this.onCreditConsumed);
  }
}
