export interface CacheUsageBucket {
  cache_read_tokens: number
  cache_write_tokens: number
  input_tokens: number
}

export interface CacheHitSummary {
  cachedTokens: number
  hitRate: number
  promptTokens: number
}

/** Returns null instead of a misleading 0% when a bucket has no prompt data. */
export function cacheHitSummary(bucket: CacheUsageBucket): CacheHitSummary | null {
  const cachedTokens = bucket.cache_read_tokens || 0
  const cacheWriteTokens = bucket.cache_write_tokens || 0
  const promptTokens = (bucket.input_tokens || 0) + cachedTokens + cacheWriteTokens

  // Session storage predates cache accounting and represents missing provider
  // fields as zeroes. Suppress an ambiguous bucket instead of charting 0%.
  if (promptTokens <= 0 || (cachedTokens === 0 && cacheWriteTokens === 0)) {
    return null
  }

  return { cachedTokens, hitRate: (cachedTokens / promptTokens) * 100, promptTokens }
}
