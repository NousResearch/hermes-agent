import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Static-analysis guard: no <button> or <Button> element in the desktop renderer
// may use the native HTML `title=` attribute. Native tooltips are unstyled,
// delayed (~500ms OS default), and visually inconsistent with the themed `Tip`.
// When a tip is warranted (see DESIGN.md — not every icon, never menu triggers),
// use `<Tip label={...}>` instead of `title=`; otherwise keep an `aria-label`.
//
// This is a source-text scan, not a behavior test — it's the same category as
// an ESLint rule, expressed as a vitest so it runs with the rest of the suite.

// Recursively walk a directory and collect all shipped .tsx file paths.
function collectTsxFiles(dir: string): string[] {
  const results: string[] = []

  for (const entry of readdirSync(dir)) {
    // Skip node_modules, dist, and __tests__ (this file itself)
    if (entry === 'node_modules' || entry === 'dist' || entry === '__tests__') {
      continue
    }

    const fullPath = join(dir, entry)
    const stat = statSync(fullPath)

    if (stat.isDirectory()) {
      results.push(...collectTsxFiles(fullPath))
    } else if (entry.endsWith('.tsx') && !entry.endsWith('.test.tsx')) {
      // Test fixtures (`<button title={title}>` stand-ins for SDK components)
      // are not shipped UI.
      results.push(fullPath)
    }
  }

  return results
}

/** Every `<button …>` / `<Button …>` opening tag in `content`, with the
 *  attribute text up to the tag's real closing `>`. */
function eachButtonOpenTag(content: string): Array<{ attrs: string; index: number; tagName: string }> {
  const tags: Array<{ attrs: string; index: number; tagName: string }> = []
  const openPattern = /<(Button|button)\b/gu
  let match: RegExpExecArray | null

  while ((match = openPattern.exec(content)) !== null) {
    const attrsStart = match.index + match[0].length
    const end = findTagClose(content, attrsStart)

    if (end < 0) {
      continue
    }

    tags.push({ attrs: content.slice(attrsStart, end), index: match.index, tagName: match[1] })
  }

  return tags
}

// The opening tag ends at the first `>` outside every `{…}` expression, string
// literal and comment. A plain `[^>]*?>` stops inside `onClick={() =>` and
// hides any `title=` written after the handler (#113688). Comments matter too:
// a `// …doesn't…` note between attributes would otherwise open a quote.
function findTagClose(content: string, start: number): number {
  let depth = 0
  let quote: null | string = null

  for (let i = start; i < content.length; i++) {
    const char = content[i]

    if (quote) {
      if (char === '\\') {
        i++
      } else if (char === quote) {
        quote = null
      }

      continue
    }

    if (char === '/' && content[i + 1] === '/') {
      const eol = content.indexOf('\n', i)

      if (eol < 0) {
        return -1
      }

      i = eol

      continue
    }

    if (char === '/' && content[i + 1] === '*') {
      const close = content.indexOf('*/', i + 2)

      if (close < 0) {
        return -1
      }

      i = close + 1

      continue
    }

    if (char === '"' || char === "'" || char === '`') {
      quote = char
    } else if (char === '{') {
      depth++
    } else if (char === '}') {
      depth--
    } else if (char === '>' && depth === 0) {
      return i
    }
  }

  return -1
}

// `title=` as an attribute of its own — not `data-title=` / `subtitle=`.
const TITLE_ATTR = /(?<![\w.-])title=/u

function titleViolations(content: string, relativePath: string): string[] {
  const violations: string[] = []

  for (const { attrs, index, tagName } of eachButtonOpenTag(content)) {
    if (TITLE_ATTR.test(attrs)) {
      const lineNum = content.slice(0, index).split('\n').length

      violations.push(`${relativePath}:${lineNum} <${tagName}> has title= — use <Tip> or aria-label`)
    }
  }

  return violations
}

describe('no native title= on button elements', () => {
  it('sees title= after inline handlers and inside comment-bearing tags, and nowhere else', () => {
    const flagged = [
      '<button onClick={() => {}} title="probe">hi</button>',
      '<Button\n  onClick={event => {\n    event.preventDefault()\n  }}\n  // macOS doesn\'t focus a button on mousedown\n  title={copy.send}\n  type="button"\n>',
      "<button className={cn(saved ? 'a' : 'b')} title={on ? m.off(t) : m.on(t)}>"
    ]

    const clean = [
      '<button aria-label="a > b" onClick={() => {}}>hi</button>',
      '<button data-title="x" onClick={() => {}}>hi</button>',
      '<button onClick={() => {}}>{`title=${x}`}</button>',
      '<span title="fine"><button type="button">hi</button></span>'
    ]

    for (const source of flagged) {
      expect(titleViolations(source, 'probe.tsx'), source).toHaveLength(1)
    }

    for (const source of clean) {
      expect(titleViolations(source, 'probe.tsx'), source).toEqual([])
    }
  })

  // Scan every shipped .tsx file under src/ for <button or <Button opening tags
  // that also carry a title= attribute (anywhere in the opening tag, which may
  // span multiple lines).
  it('uses <Tip> or aria-label instead of native title= on all button elements', () => {
    const violations: string[] = []
    const srcDir = resolve(__dirname, '../../..')

    for (const filePath of collectTsxFiles(srcDir)) {
      const content = readFileSync(filePath, 'utf-8')
      const relativePath = filePath.replace(srcDir + '/', '')

      violations.push(...titleViolations(content, relativePath))
    }

    expect(violations, violations.join('\n')).toEqual([])
  })
})
