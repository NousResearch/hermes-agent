import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Static-analysis guard: no <button> or <Button> element in the desktop renderer
// may use the native HTML `title=` attribute. Native tooltips are unstyled,
// delayed (~500ms OS default), and visually inconsistent with the themed `Tip`.
// When a tip is warranted (see DESIGN.md — not every icon, never menu triggers),
// use `<Tip label={...}>` instead of `title=`.
//
// This is a source-text scan, not a behavior test — it's the same category as
// an ESLint rule, expressed as a vitest so it runs with the rest of the suite.

// Recursively walk a directory and collect all .tsx file paths.
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
    } else if (entry.endsWith('.tsx')) {
      results.push(fullPath)
    }
  }

  return results
}

describe('no native title= on button elements', () => {
  // Scan every .tsx file under src/ for <button or <Button opening tags that
  // also carry a title= attribute (anywhere in the opening tag, which may span
  // multiple lines).
  it('uses <Tip> instead of native title= on all button elements', () => {
    const violations: string[] = []
    const srcDir = resolve(__dirname, '../../..')

    for (const filePath of collectTsxFiles(srcDir)) {
      const content = readFileSync(filePath, 'utf-8')
      const relativePath = filePath.replace(srcDir + '/', '')

      // Match <Button ...> or <button ...> opening tags (may span multiple lines).
      // Scan to the brace-depth-0 `>` so a `>` inside a JSX expression (e.g.
      // `onClick={() => ...}`) doesn't truncate the tag and hide a later
      // title= (#113688). String literals are skipped the same way.
      for (const { tagName, attrs, index } of eachButtonOpenTag(content)) {
        if (/\btitle=/.test(attrs)) {
          const lineNum = content.slice(0, index).split('\n').length
          violations.push(`${relativePath}:${lineNum} <${tagName}> has title= — use <Tip>`)
        }
      }
    }

    expect(violations, violations.join('\n')).toEqual([])
  })
})

function eachButtonOpenTag(content: string): Array<{ tagName: string; attrs: string; index: number }> {
  const tags: Array<{ tagName: string; attrs: string; index: number }> = []
  const openPattern = /<(Button|button)\b/gsu
  let match: RegExpExecArray | null

  while ((match = openPattern.exec(content)) !== null) {
    const end = findBraceDepthZeroClose(content, match.index + match[0].length)
    if (end < 0) {
      continue
    }
    tags.push({ tagName: match[1], attrs: content.slice(match.index, end), index: match.index })
  }

  return tags
}

function findBraceDepthZeroClose(content: string, start: number): number {
  let depth = 0
  let quote: string | null = null

  for (let i = start; i < content.length; i++) {
    const char = content[i]
    if (quote) {
      if (char === quote && content[i - 1] !== '\\') {
        quote = null
      }
      continue
    }
    if (char === '"' || char === "'" || char === '`') {
      quote = char
      continue
    }
    if (char === '{') {
      depth++
    } else if (char === '}') {
      depth--
    } else if (char === '>' && depth === 0) {
      return i
    }
  }

  return -1
}
