// Render cache for mermaid diagrams. Transcript remounts (history reloads,
// file previews, tab switches) used to re-run mermaid.parse+layout for every
// fence; with several diagrams that froze the renderer for noticeable stretches.
// Completed SVGs are kept per (source, theme) and reused across mounts, in-flight
// renders for identical pairs are shared, misses are serialized so diagrams do
// not monopolise one event-loop turn, and the heavy mermaid runtime is imported
// inside the render callback — after the source fallback has painted.

type Theme = 'dark' | 'default'

interface CacheEntry {
  svg: string
}

const MAX_CACHED_SVGS = 64

const completed = new Map<string, CacheEntry>()
const inFlight = new Map<string, Promise<string>>()

// Serialized misses: one diagram renders at a time, next starts only after the
// previous settled. This keeps a transcript full of diagrams from turning into
// one long uninterruptible block of layout work.
let queueTail: Promise<void> = Promise.resolve()

export function resetMermaidRenderCacheForTests(): void {
  completed.clear()
  inFlight.clear()
  queueTail = Promise.resolve()
}

export function cachedMermaidSvgCount(): number {
  return completed.size
}

function cacheKey(source: string, theme: Theme): string {
  // JSON-encoded pair cannot collide across different (source, theme) combos.
  return JSON.stringify([theme, source])
}

function evictIfNeeded(): void {
  while (completed.size > MAX_CACHED_SVGS) {
    // Map preserves insertion order; the oldest entry is the least recently
    // inserted render. Re-insertion on cache hit would be needed for true LRU;
    // transcript diagrams are re-read far more often than they are added, so
    // FIFO keeps the common case (scroll back through history) fully cached.
    const oldest = completed.keys().next().value

    if (oldest === undefined) {
      return
    }

    completed.delete(oldest)
  }
}

async function runSerialized<T>(task: () => Promise<T>): Promise<T> {
  const run = queueTail.then(task, task)
  queueTail = run.then(
    () => undefined,
    () => undefined
  )

  return run
}

export interface MermaidRenderResult {
  svg: string
}

async function renderUncached(source: string, theme: Theme): Promise<string> {
  // The import stays inside the render path on purpose: the lazy embed chunk
  // stays lightweight and the mermaid runtime is only evaluated once a diagram
  // actually needs rendering, after the source fallback has painted.
  const { default: mermaid } = await import('mermaid')
  mermaid.initialize({
    fontFamily: 'inherit',
    securityLevel: 'strict',
    startOnLoad: false,
    theme
  })
  const id = `mmd-${Math.random().toString(36).slice(2)}`
  const result = await mermaid.render(id, source)

  return result.svg
}

export function renderMermaidSvg(
  source: string,
  theme: Theme
): Promise<MermaidRenderResult> {
  const key = cacheKey(source, theme)

  const hit = completed.get(key)

  if (hit) {
    return Promise.resolve(hit)
  }

  const existing = inFlight.get(key)

  if (existing) {
    return existing.then((svg) => ({ svg }))
  }

  const render = runSerialized(() => renderUncached(source, theme))
    .then((svg) => {
      completed.set(key, { svg })
      evictIfNeeded()
      inFlight.delete(key)

      return svg
    })
    .catch((error: unknown) => {
      // Failed entries never enter the cache, so a transient parse error (or a
      // partially streamed diagram re-rendered after an edit) can retry.
      inFlight.delete(key)
      throw error
    })

  inFlight.set(key, render)

  return render.then((svg) => ({ svg }))
}

// The embed defers the first uncached render by one frame so the source
// fallback paints before the CPU-heavy mermaid chunk is imported and parsed.
export function nextPaint(): Promise<void> {
  return new Promise((resolve) => {
    if (typeof requestAnimationFrame === 'function') {
      requestAnimationFrame(() => resolve())
    } else {
      setTimeout(resolve, 0)
    }
  })
}
