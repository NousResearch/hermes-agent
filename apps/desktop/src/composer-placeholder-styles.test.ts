// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const STYLES = readFileSync(join(dirname(fileURLToPath(import.meta.url)), 'styles.css'), 'utf8').replace(
  /\/\*[\s\S]*?\*\//g,
  ''
)

describe('composer placeholder', () => {
  // Chromium exposes ::before text as the contenteditable's AXValue unless the
  // content has alt text. Dictation tools then treat the hint as typed text.
  it('paints the hint with empty alt text so it stays out of the field value', () => {
    const rule = STYLES.match(/\[data-slot='composer-rich-input'\]:is\(:empty, \[data-empty\]\)::before\s*\{([^}]*)\}/)

    expect(rule).not.toBeNull()
    expect(rule![1]).toMatch(/content:\s*attr\(data-placeholder\)\s*\/\s*(''|"")\s*;/)
  })
})
