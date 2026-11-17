/**
 * Regression guard: the desktop's gateway-switch publication runs inside
 * nanostores' batch(). nanostores 1.4.0–1.4.1 annotated batch() with
 * `/* @__NO_SIDE_EFFECTS__ *\/`, which let production bundlers (Rollup, via
 * `vite build`) erase result-unused batch(...) calls — callback included —
 * deleting the entire publication from release builds and freezing
 * profile/agent switching there. Dev builds and vitest run unminified, so
 * only the packaged app broke. nanostores 1.4.2 removed the annotation;
 * this test fails if the installed version ever carries it again.
 */
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import path from 'node:path'

import { describe, expect, it } from 'vitest'

const require = createRequire(import.meta.url)

describe('nanostores batch()', () => {
  it('is not annotated @__NO_SIDE_EFFECTS__ — bundlers would erase its calls', () => {
    const nanostoresEntry = require.resolve('nanostores')
    const atomSource = readFileSync(path.join(path.dirname(nanostoresEntry), 'atom/index.js'), 'utf8')

    // Capture any block comment immediately preceding the batch export.
    const match = atomSource.match(/(\/\*(?:[^*]|\*(?!\/))*\*\/\s*)?export const batch\b/)

    expect(match, 'nanostores atom/index.js should export const batch').toBeTruthy()
    expect(match![1] ?? '').not.toContain('__NO_SIDE_EFFECTS')
    expect(match![1] ?? '').not.toContain('__PURE__')
  })
})
