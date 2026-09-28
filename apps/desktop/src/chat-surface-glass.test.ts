// @vitest-environment node
import fs from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const SRC = dirname(fileURLToPath(import.meta.url))
const stylesheet = fs.readFileSync(resolve(SRC, 'styles.css'), 'utf8')

/** The `selector { body }` block whose selector mentions the transcript frame. */
const transcriptRule = (() => {
  for (const [, selector, body] of stylesheet.matchAll(/([^{}]*)\{([^{}]*)\}/g)) {
    if (selector.includes('[data-chat-transcript-frame]')) {
      return { body, selector }
    }
  }

  return null
})()

describe('glass chat transcript surface', () => {
  it('owns a tinted, framed and blurred fill instead of exposing the window backdrop', () => {
    expect(transcriptRule).not.toBeNull()
    const { body, selector } = transcriptRule!

    expect(body).toMatch(/background:\s*color-mix\([^;]+var\(--ui-bg-chrome\)\s+(?:[7-9]\d|100)%/)
    expect(body).not.toMatch(/background(?:-color)?:\s*transparent/)
    expect(body).toMatch(/border:\s*1px\s+solid\s+var\(--ui-stroke-tertiary\)/)
    expect(body).toMatch(/border-radius:\s*var\(--radius-/)
    expect(body).toMatch(/backdrop-filter:\s*blur\(/)
    expect(body).toMatch(/box-shadow:[^;]*inset\s+0\s+1px\s+0/)
    expect(selector).toContain(':root[data-hermes-glass]')
  })

  it('frames the transcript in clear mode too, where native opacity also reveals the desktop', () => {
    expect(transcriptRule?.selector).toContain(':root[data-hermes-clear]')
  })

  it('does not change the global glass painter or terminal surface contract', () => {
    const rootGlassRule = stylesheet.match(/:root\[data-hermes-glass\]\s*\{([^}]*)\}/)?.[1]

    expect(rootGlassRule).toContain('--ui-chat-surface-background: transparent')
    expect(rootGlassRule).toContain('--ui-terminal-surface-background: var(--ui-bg-chrome)')
  })
})
