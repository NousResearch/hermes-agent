'use client'

/**
 * The ONLY static importer of `react-shiki` (and through it the multi-MB
 * shiki language/theme bundle). Every consumer reaches this module through
 * `React.lazy(() => import('./shiki-block'))` — see `LazyShiki` in
 * shiki-highlighter.tsx — so the shiki chunk stays entirely off the
 * cold-start path and loads on the first highlighted code block instead.
 *
 * Do NOT import this module statically from anything the entry graph
 * reaches, or the chunk moves back into boot.
 */
import ShikiHighlighter from 'react-shiki'

export default ShikiHighlighter
