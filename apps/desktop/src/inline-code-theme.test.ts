// @vitest-environment node
import { readFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

// Static-analysis guard: system-message markdown (async-delegation / cron
// results) renders `.aui-md :not(pre) > code` without the assistant-content
// slot. Inline code must still pick up the theme tokens, or Tailwind
// Typography's fixed near-black ink wins on every dark theme.
//
// Source-text scan of styles.css — same category as no-native-title.test.ts.

const SRC = dirname(fileURLToPath(import.meta.url))
const STYLES = resolve(SRC, 'styles.css')

const INLINE_CODE = /:not\(\s*pre\s*\)\s*>\s*code/
const AUI_MD = /\.aui-md\b/
const ASSISTANT_SLOT = /\[\s*data-slot\s*=\s*['"]aui_assistant-message-content['"]\s*\]/
const SYSTEM_SLOT = /aui_system-message-root/
const HEX_COLOR = /#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\b/

function stripComments(css: string): string {
  return css.replace(/\/\*[\s\S]*?\*\//g, '')
}

function extractRules(css: string): Array<{ selector: string; body: string }> {
  const rules: Array<{ selector: string; body: string }> = []
  let lastBoundary = 0

  for (let i = 0; i < css.length; i++) {
    const ch = css[i]
    if (ch === '{') {
      const selector = css.slice(lastBoundary, i).trim()
      let depth = 1
      let j = i + 1
      while (j < css.length && depth > 0) {
        if (css[j] === '{') depth++
        else if (css[j] === '}') depth--
        j++
      }
      rules.push({ selector, body: css.slice(i + 1, j - 1) })
      lastBoundary = i + 1
    } else if (ch === '}') {
      lastBoundary = i + 1
    }
  }

  return rules
}

function splitSelectorList(selector: string): string[] {
  const parts: string[] = []
  let current = ''
  let depth = 0

  for (const ch of selector) {
    if (ch === '(') depth++
    else if (ch === ')') depth--
    if (ch === ',' && depth === 0) {
      parts.push(current.trim())
      current = ''
      continue
    }
    current += ch
  }

  if (current.trim()) parts.push(current.trim())
  return parts
}

function isInlineCodeTokenRule(rule: { selector: string; body: string }): boolean {
  return INLINE_CODE.test(rule.selector) && rule.body.includes('--ui-inline-code-foreground')
}

function coversMarkdownSurface(selector: string): boolean {
  return splitSelectorList(selector).some(part => {
    if (!INLINE_CODE.test(part)) return false
    if (SYSTEM_SLOT.test(part)) return true
    return AUI_MD.test(part) && !ASSISTANT_SLOT.test(part)
  })
}

describe('inline code theme tokens on markdown surfaces', () => {
  const css = stripComments(readFileSync(STYLES, 'utf8'))
  const tokenRules = extractRules(css).filter(isInlineCodeTokenRule)

  it('applies inline-code tokens to .aui-md without requiring the assistant slot', () => {
    const covering = tokenRules.filter(rule => coversMarkdownSurface(rule.selector))

    expect(
      covering.length,
      'expected a :not(pre) > code rule that covers .aui-md (or aui_system-message-root) without requiring [data-slot=aui_assistant-message-content]'
    ).toBeGreaterThan(0)

    for (const rule of covering) {
      expect(rule.body).toMatch(/background\s*:\s*var\(\s*--ui-inline-code-background\s*\)/)
      expect(rule.body).toMatch(/color\s*:\s*var\(\s*--ui-inline-code-foreground\s*\)/)
      expect(rule.body, 'inline-code rule must not hard-code hex colors').not.toMatch(HEX_COLOR)
    }
  })
})
