import { Component, type ReactNode } from 'react'

// Chromium/Electron: "TypeError: Failed to fetch dynamically imported
// module: …". Vite also emits "error loading a dynamically imported module",
// Safari "a module that couldn't be instantiated" — all three are
// transport/packaging failures (missing chunk in the bundle, asar/asar.unpacked
// mismatch, install mid-update), never code bugs. Narrow to every spelling so
// a genuine render throw is never claimed as a chunk failure.
const CHUNK_LOAD_ERROR =
  /Failed to fetch dynamically imported module|error loading a dynamically imported module|Dynamically imported module .*couldn't be instantiated/u

/** A failed lazy-chunk fetch (missing/broken bundle piece), not a code bug. */
export function isChunkLoadError(error: unknown): boolean {
  return error instanceof Error && CHUNK_LOAD_ERROR.test(error.message)
}

interface ShikiChunkBoundaryProps {
  children: ReactNode
  /** Rendered in place of the lazy subtree when its chunk fails to load. */
  fallback: ReactNode
}

/**
 * Error boundary for a lazily-imported renderer whose FAILURE TO LOAD must
 * degrade to plain content instead of unwinding (#95995). React.Suspense
 * covers only the pending import; a rejection re-throws at render and, for
 * chat code fences, reached `markdown-render` (markdown-text.tsx), whose
 * HugeTextFallback collapsed the WHOLE reply into a raw-Markdown panel
 * because one fence couldn't load its highlighter — 229 such catches in one
 * reporter's desktop.log, all `Failed to fetch dynamically imported module:
 * …shiki-block-Dcm1B2nM.js`.
 *
 * Same re-throw pattern as MessageRenderBoundary: it swallows ONLY the
 * chunk-load class. Any other render error re-throws and keeps its trip to
 * the message-level boundary, logged to desktop.log with a component stack —
 * exactly as before this boundary existed.
 */
export class ShikiChunkBoundary extends Component<ShikiChunkBoundaryProps, { error: unknown | null }> {
  state: { error: unknown | null } = { error: null }

  static getDerivedStateFromError(error: unknown) {
    return { error }
  }

  componentDidCatch(error: unknown) {
    if (isChunkLoadError(error)) {
      // One warn per caught block keeps the failure diagnosable. The
      // degraded render below is the handled outcome, not a crash, so this
      // stays a warn — the reporter's 229-error session log was the problem
      // this fix exists to make survivable.
      console.warn('[shiki-block] highlighter chunk failed to load; rendering code unhighlighted', error)
    }
  }

  render() {
    const { error } = this.state

    if (error === null) {
      return this.props.children
    }

    // Only the chunk-load failure is ours to degrade; a real render bug
    // must still reach `markdown-render`.
    if (!isChunkLoadError(error)) {
      throw error
    }

    return this.props.fallback
  }
}
