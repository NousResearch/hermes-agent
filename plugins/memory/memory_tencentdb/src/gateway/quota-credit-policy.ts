/**
 * Credit 上报策略 —— 决定内核 L1/L2/L3 抽取完成时上报给 UpdateMemoryPlusUsage
 * 的 CreditDelta 数值。
 *
 * 背景：当 llm.provider="proxy" 时，context_proxy 已经在 LLM 应答完成后向同一
 * 个 UpdateMemoryPlusUsage 上报了一次 CreditDelta = total_tokens（见
 * context_proxy/src/credit-reporter.ts）。如果内核再上报一次同一次调用的 token，
 * 会导致重复计费。
 *
 * 策略：
 *   - provider="openai"：原样返回 —— 内核是唯一 credit 上报方
 *   - provider="proxy": 返回 0 —— proxy 已经报过，内核跳过
 *
 * memoryDelta 与此策略无关：memory 是内核独有的语义（写入了几条记忆），
 * proxy 完全不知情，必须由内核负责上报。
 */

export type LlmProvider = "openai" | "proxy";

export function shouldSkipCreditReport(provider: LlmProvider | undefined): boolean {
  return (provider ?? "openai") === "proxy";
}

/**
 * 计算实际应该上报的 CreditDelta。
 *
 * @param rawCreditUsed  内核 llmRunner.accumulatedCredit 累积的 token 数
 * @param provider       llm.provider
 * @returns 应该上报的 CreditDelta（provider=proxy 时恒为 0）
 */
export function resolveReportedCredit(
  rawCreditUsed: number,
  provider: LlmProvider | undefined,
): number {
  return shouldSkipCreditReport(provider) ? 0 : rawCreditUsed;
}
