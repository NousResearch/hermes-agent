import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

const styles = readFileSync(resolve(process.cwd(), 'src/styles.css'), 'utf8')

describe('resolved bidi surface styles', () => {
  it('isolates the resolved direction instead of re-detecting first-strong text', () => {
    expect(styles).toMatch(
      /\[data-slot='aui_assistant-message-content'\][\s\S]*?\[data-slot='composer-rich-input'\]\s*\{\s*unicode-bidi:\s*isolate;\s*text-align:\s*start;\s*\}/
    )
  })
})
