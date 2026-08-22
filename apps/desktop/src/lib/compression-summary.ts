/** Synthetic user rows emitted by Hermes context-compaction engines. */
const COMPRESSION_SUMMARY_PREFIXES = [
  '[CONTEXT COMPACTION',
  '[CONTEXT SUMMARY]',
  '[Recent Summary',
  '[Session Arc Summary'
] as const

export function isCompressionSummaryText(text: string): boolean {
  const normalized = text.trimStart()

  return COMPRESSION_SUMMARY_PREFIXES.some(prefix => normalized.startsWith(prefix))
}
