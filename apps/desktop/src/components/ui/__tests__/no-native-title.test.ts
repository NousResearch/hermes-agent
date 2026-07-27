import { readdirSync, readFileSync, statSync } from 'node:fs'
import { join, resolve } from 'node:path'

import { describe, expect, it } from 'vitest'

// Static-analysis guard: no <button> or <Button> element in the desktop renderer
// may use the native HTML `title=` attribute. Native tooltips are unstyled,
// delayed (~500ms OS default), and visually inconsistent with the app's instant
// themed `Tip` component. Use `<Tip label={...}>` instead.
//
// This is a source-text scan, not a behavior test — it's the same category as
// an ESLint rule, expressed as a vitest so it runs with the rest of the suite.
// See DESIGN.md "Buttons — one component" for the rule.

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
    const srcDir = resolve(__dirname, '../..')

    for (const filePath of collectTsxFiles(srcDir)) {
      const content = readFileSync(filePath, 'utf-8')
      const relativePath = filePath.replace(srcDir + '/', '')

      // Match <Button ...> or <button ...> opening tags (may span multiple lines).
      // We use a non-greedy match up to the closing > — this won't perfectly
      // handle every edge case (e.g. > inside a string literal), but it's good
      // enough for a lint-style guard.
      const tagPattern = /<(Button|button)\b([^>]*?)>/gsu
      let match: RegExpExecArray | null

      while ((match = tagPattern.exec(content)) !== null) {
        const tagName = match[1]
        const attrs = match[2]

        if (/\btitle=/.test(attrs)) {
          const lineNum = content.slice(0, match.index).split('\n').length
          violations.push(`${relativePath}:${lineNum} <${tagName}> has title= — use <Tip>`)
        }
      }
    }

    expect(violations, violations.join('\n')).toEqual([])
  })
})
