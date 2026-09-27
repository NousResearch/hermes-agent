// @vitest-environment node
import fs from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))
const stylesheet = fs.readFileSync(resolve(SRC, 'styles.css'), 'utf8')
const glassTranscriptRule = stylesheet.match(
  /:root\[data-hermes-glass\]\s+\[data-chat-transcript-frame\]\s*\{([^}]*)\}/
)?.[1]

describe('glass chat transcript surface', () => {
  it('owns a tinted, framed and blurred fill instead of exposing the window backdrop', () => {
    expect(glassTranscriptRule).toBeDefined()
    expect(glassTranscriptRule).toMatch(/background:\s*color-mix\([^;]+var\(--ui-bg-chrome\)\s+(?:[7-9]\d|100)%/)
    expect(glassTranscriptRule).not.toMatch(/background(?:-color)?:\s*transparent/)
    expect(glassTranscriptRule).toMatch(/border:\s*1px\s+solid\s+var\(--ui-stroke-tertiary\)/)
    expect(glassTranscriptRule).toMatch(/border-radius:\s*var\(--radius-/)
    expect(glassTranscriptRule).toMatch(/backdrop-filter:\s*blur\(/)
  })

  it('does not change the global glass painter or terminal surface contract', () => {
    const rootGlassRule = stylesheet.match(/:root\[data-hermes-glass\]\s*\{([^}]*)\}/)?.[1]

    expect(rootGlassRule).toContain('--ui-chat-surface-background: transparent')
    expect(rootGlassRule).toContain('--ui-terminal-surface-background: var(--ui-bg-chrome)')
  })
})
