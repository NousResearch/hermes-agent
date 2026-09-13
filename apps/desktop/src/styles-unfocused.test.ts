import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const css = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'styles.css'), 'utf8')

describe('unfocused chat surface (#101890)', () => {
  it('dims with opacity and does not grayscale the streaming layer', () => {
    const block = css.split('[data-chat-surface][data-chat-unfocused]')[1]?.split('[data-chat-surface]')[0] ?? ''

    expect(block).toContain('opacity: var(--chat-unfocused-opacity)')
    expect(block).not.toContain('grayscale')
  })
})
