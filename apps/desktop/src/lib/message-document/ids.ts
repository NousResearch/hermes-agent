/** Stable token ids — survive path rewrites after attachment sync. */
export function createTokenId(prefix: 'text' | 'attachment'): string {
  const cryptoRef = typeof crypto !== 'undefined' ? crypto : undefined

  if (cryptoRef?.randomUUID) {
    return `${prefix}-${cryptoRef.randomUUID()}`
  }

  return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2, 10)}`
}
