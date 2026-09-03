/**
 * Regression: the desktop main process must finish initializing rememberLog()
 * state before any top-level statement that logs.
 *
 * `readPersistedPoolLimits()` runs at module top level and logs on BOTH code
 * paths (persisted file / env-var-or-defaults fallback). `rememberLog()`
 * pushes onto `hermesLog` and appends to `desktopLogBuffer`, so those
 * declarations must execute first. When they were declared ~110 lines below
 * the top-level call, every desktop launch died during ESM module evaluation
 * with `TypeError: Cannot read properties of undefined (reading 'push')` —
 * esbuild lowers `const`/`let` TDZ to `undefined` instead of throwing, so the
 * packaged app crashed at `hermesLog.push(...)` and the window never appeared
 * (#101941).
 *
 * main.ts cannot be imported in unit tests (it needs Electron's `app` at
 * module scope), so this locks the source-level ordering contract instead:
 * the log-state declarations exist exactly once and precede the top-level
 * `readPersistedPoolLimits()` call.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const MAIN_TS = path.join(__dirname, 'main.ts')

function mainSource(): string {
  return fs.readFileSync(MAIN_TS, 'utf8')
}

// Returns the index of `needle`, asserting it appears exactly once so a
// pasted duplicate cannot silently satisfy the ordering check.
function uniqueIndexOf(source: string, needle: string): number {
  const first = source.indexOf(needle)
  assert.notStrictEqual(first, -1, `expected exactly one occurrence of ${needle}`)

  assert.strictEqual(
    source.indexOf(needle, first + 1),
    -1,
    `expected exactly one occurrence of ${needle}`
  )

  return first
}

test('log buffer state is declared before the top-level readPersistedPoolLimits() call', () => {
  const source = mainSource()

  const poolLimitsCall = uniqueIndexOf(source, 'let poolLimits = readPersistedPoolLimits()')

  for (const declaration of [
    'const hermesLog = []',
    "let desktopLogBuffer = ''",
    'let desktopLogFlushTimer = null',
    'let desktopLogFlushPromise = Promise.resolve()'
  ]) {
    const at = uniqueIndexOf(source, declaration)

    assert.ok(
      at < poolLimitsCall,
      `${declaration} must be declared before \`let poolLimits = readPersistedPoolLimits()\` — rememberLog() touches it during module evaluation (#101941)`
    )
  }
})
