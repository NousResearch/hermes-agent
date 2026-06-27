import type { ModelCapabilities } from '@/types/hermes'

export interface ModelReasoningPolicy {
  configurable: boolean
  defaultEffort: string
  effort: string
  label: string
}

/** Resolve one picker row's effective reasoning policy.
 *
 * Built-in/catalog models retain the historical behavior (reasoning support
 * implies an adjustable control with medium as the default). Configured proxy
 * models can instead declare an explicit default or a fixed policy label.
 */
export function resolveModelReasoningPolicy(
  capabilities: ModelCapabilities | undefined,
  options: { currentEffort: string; isCurrent: boolean; presetEffort?: string }
): ModelReasoningPolicy {
  const supported = capabilities?.reasoning ?? true
  const configurable = capabilities?.reasoning_configurable ?? supported
  const defaultEffort = capabilities?.reasoning_default?.trim() || 'medium'
  const effort = (options.isCurrent ? options.currentEffort : options.presetEffort)?.trim() || defaultEffort
  const label = capabilities?.reasoning_label?.trim() || ''

  return { configurable, defaultEffort, effort, label }
}
