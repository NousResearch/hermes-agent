import { act, cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

// Regression coverage for #95995: a failed dynamic import of the lazily-loaded
// Shiki code-block chunk (missing from a packaged app bundle — the reporter's
// desktop.log shows 229× `Failed to fetch dynamically imported module:
// …shiki-block-Dcm1B2nM.js`) rejects the `React.lazy()` promise. React.Suspense
// only covers the *pending* state, so the rejection threw past it to the
// `markdown-render` boundary, whose fallback collapses the WHOLE reply into
// the HugeTextFallback raw-Markdown panel. The block-local ShikiChunkBoundary
// now degrades just the fence to plain unhighlighted code.
//
// The mock resolves to a component that throws the fetch error during render —
// the same way React surfaces a rejected lazy payload (a rejected import
// re-throws at render time). Do NOT throw inside the factory itself: a
// throwing factory leaves rejected promises in the vitest mocker registry,
// and under CI load one escapes as an "unhandled error during the test run"
// attributed to whichever sibling test file the worker is running (#94415).
const { chunkError } = vi.hoisted(() => ({
  chunkError: new Error(
    'Failed to fetch dynamically imported module: file:///Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/assets/shiki-block-Dcm1B2nM.js'
  )
}))

vi.mock('./shiki-block', () => ({
  default: () => {
    throw chunkError
  }
}))

import { MarkdownTextContent } from '@/components/assistant-ui/markdown-text'

import { LazyShiki } from './shiki-highlighter'

afterEach(cleanup)

const MESSAGE = [
  '# Deploy checklist', '',
  'Run the migration before swapping traffic:', '',
  '```bash',
  'psql -f migrate.sql',
  '```', '',
  'Then watch the logs until the pool is warm.'
].join('\n')

// Renders through the REAL message pipeline (Streamdown + the markdown-render
// boundary), the way the packaged app hit it: prose renders rich, and only the
// code block degrades to plain text — no HugeTextFallback panel.
describe('a reply survives a failed shiki-block chunk load (#95995)', () => {
  it('keeps prose rendered and shows the code block as plain text', async () => {
    // React 19 reports boundary-caught errors through console.error in dev —
    // silence that; what matters is what the boundary logged and rendered.
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => undefined)

    try {
      const { container } = render(<MarkdownTextContent isRunning={false} text={MESSAGE} />)

      // The rejection settles a tick after the initial Suspense-pending render
      // (which coincidentally shows the same plain text already) — give it
      // real time to propagate before asserting nothing regressed.
      await act(() => new Promise(resolve => setTimeout(resolve, 300)))

      // Prose is still rendered rich: a real heading, not raw markdown inside
      // the bordered font-mono HugeTextFallback panel (which renders no <h1>).
      expect(container.querySelector('h1')?.textContent).toBe('Deploy checklist')
      expect(container.textContent).toContain('Run the migration')
      // The raw fence marker is gone — markdown parsed, not passthrough.
      expect(container.textContent).not.toContain('```bash')

      // The fence degraded to plain code: content present, still inside the
      // code card the app renders before highlight lands.
      expect(container.querySelector('[data-slot="code-card"]')).not.toBeNull()
      expect(container.textContent).toContain('psql -f migrate.sql')

      // The failure stays diagnosable: one warn per caught block, not the
      // reporter's 229-error storm.
      expect(warnSpy).toHaveBeenCalledTimes(1)
    } finally {
      errorSpy.mockRestore()
      warnSpy.mockRestore()
    }
  })

  it('re-throws a genuine render error so markdown-render still catches it', () => {
    // Same mock, different error: only the chunk-load failure class is ours to
    // degrade. A real render bug must keep its trip to the message-level
    // boundary (logged with a component stack, HugeTextFallback if it throws
    // again), exactly as before this boundary existed.
    chunkError.message = 'grammar tokenizer exploded'

    const spy = vi.spyOn(console, 'error').mockImplementation(() => undefined)

    try {
      expect(() => render(<LazyShiki code="const a = 1" language="typescript" />)).toThrow(
        'grammar tokenizer exploded'
      )
    } finally {
      spy.mockRestore()
      chunkError.message =
        'Failed to fetch dynamically imported module: file:///Applications/Hermes.app/Contents/Resources/app.asar.unpacked/dist/assets/shiki-block-Dcm1B2nM.js'
    }
  })
})
