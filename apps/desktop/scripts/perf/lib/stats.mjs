// Shared numeric helpers for perf scenarios. Every measure-*/profile-* script
// used to carry its own copy of these.

/** Nearest-rank percentile over an UNSORTED array. p in [0,1]. */
export function percentile(values, p) {
  if (!values.length) {
    return 0
  }

  const sorted = [...values].sort((a, b) => a - b)
  const idx = Math.min(sorted.length - 1, Math.floor(sorted.length * p))

  return sorted[idx]
}

/** min/p50/p90/p95/p99/max/mean over a sample array (rounded to 2dp). */
export function summarize(values) {
  const round = n => Math.round(n * 100) / 100

  if (!values.length) {
    return { n: 0, min: 0, p50: 0, p90: 0, p95: 0, p99: 0, max: 0, mean: 0 }
  }

  const sorted = [...values].sort((a, b) => a - b)
  const mean = values.reduce((a, b) => a + b, 0) / values.length

  return {
    n: values.length,
    min: round(sorted[0]),
    p50: round(percentile(sorted, 0.5)),
    p90: round(percentile(sorted, 0.9)),
    p95: round(percentile(sorted, 0.95)),
    p99: round(percentile(sorted, 0.99)),
    max: round(sorted[sorted.length - 1]),
    mean: round(mean)
  }
}

/** Median of a numeric array (used to reduce N repeated runs to one number). */
export function median(values) {
  return percentile(values, 0.5)
}

/** Frame-interval histogram matching the buckets the stream scripts reported. */
export function frameHistogram(frames) {
  const buckets = { '<=16.7': 0, '16.7-33': 0, '33-50': 0, '50-100': 0, '100-200': 0, '>200': 0 }

  for (const f of frames) {
    if (f <= 16.7) buckets['<=16.7']++
    else if (f <= 33) buckets['16.7-33']++
    else if (f <= 50) buckets['33-50']++
    else if (f <= 100) buckets['50-100']++
    else if (f <= 200) buckets['100-200']++
    else buckets['>200']++
  }

  return buckets
}

/**
 * Rank functions by self-time from a V8 CPU profile (Profiler.stop output).
 * Returns the top `limit` entries as { ms, name, url, line }.
 */
export function cpuProfileTopSelf(profile, limit = 30) {
  const samples = profile.samples || []
  const timeDeltas = profile.timeDeltas || []
  const nodes = new Map(profile.nodes.map(n => [n.id, n]))
  const selfUs = new Map()

  for (let i = 0; i < samples.length; i++) {
    const id = samples[i]
    selfUs.set(id, (selfUs.get(id) || 0) + (timeDeltas[i] ?? 0))
  }

  return [...selfUs.entries()]
    .map(([id, us]) => {
      const cf = nodes.get(id)?.callFrame || {}

      return {
        ms: us / 1000,
        name: cf.functionName || '(anonymous)',
        url: String(cf.url || '').slice(-70),
        line: cf.lineNumber
      }
    })
    .filter(x => !/\(root\)|\(idle\)|\(garbage collector\)|\(program\)/.test(x.name))
    .sort((a, b) => b.ms - a.ms)
    .slice(0, limit)
}
