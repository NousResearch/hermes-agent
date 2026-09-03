import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'

import { build } from 'esbuild'
import { test } from 'vitest'

// Regression test for #102018 / fix 05f548f35d:
//
// readPersistedPoolLimits() runs at module evaluation and synchronously
// calls rememberLog() on every code path. rememberLog() does
// `hermesLog.push(...)` (and reads desktopLogBuffer / desktopLogFlushTimer /
// desktopLogFlushPromise). esbuild lowers `const`/`let` to `var`, so if any
// of those four state declarations is lexically *after* the top-level
// `var poolLimits = readPersistedPoolLimits()` in the bundled output, the
// binding exists but is `undefined` at the time of the call — and Electron
// crashes at launch with `TypeError: Cannot read properties of undefined
// (reading 'push')`.
//
// Source-level ordering in main.ts does NOT guarantee bundle-level ordering:
// esbuild hoists standalone `var x = literal` initialisers into a single
// block at the top of the module, while `var x = someCall()` (where the
// right-hand side is a function call) stays at its original lexical
// position. Asserting the source order would have hidden this bug.
//
// The test therefore bundles main.ts the same way scripts/bundle-electron-
// main.mjs does, then asserts the four state lines each appear *before*
// the `readPersistedPoolLimits` call line in the bundled output.

const APPS_DESKTOP = resolve(__dirname, '..')
const MAIN_ENTRY = resolve(APPS_DESKTOP, 'electron/main.ts')

const STATE_LINE_PATTERNS = [
  /^var hermesLog = \[\];?$/,
  /^var desktopLogBuffer = "";?$/,
  /^var desktopLogFlushTimer = null;?$/,
  /^var desktopLogFlushPromise = Promise\.resolve\(\);?$/
]

const CALL_LINE_PATTERN = /^var poolLimits = readPersistedPoolLimits\(\);?$/

async function bundleMain(): Promise<string> {
  const result = await build({
    entryPoints: [MAIN_ENTRY],
    bundle: true,
    platform: 'node',
    format: 'esm',
    target: 'node22',
    // Match scripts/bundle-electron-main.mjs externals — these are provided
    // by the Electron runtime / staged separately and must not be inlined.
    external: ['electron', 'node-pty', 'get-windows', 'fs'],
    write: false,
    logLevel: 'silent'
  })

  return result.outputFiles[0].text
}

test('main.ts bundles with all rememberLog state declared before the pool-limits call (#102018)', async () => {
  const out = await bundleMain()
  const lines = out.split('\n')

  const stateLineNumbers = STATE_LINE_PATTERNS.map((re) =>
    lines.findIndex((l) => re.test(l.trim()))
  )

  const callLineNumber = lines.findIndex((l) => CALL_LINE_PATTERN.test(l.trim()))

  // Sanity: every expected state line must be present in the bundle. If
  // any of these is -1 the bundle was restructured in a way the test
  // doesn't recognise; fail loudly so a human can update the patterns
  // rather than silently passing.
  for (const [i, n] of stateLineNumbers.entries()) {
    assert.notEqual(
      n,
      -1,
      `expected state line ${STATE_LINE_PATTERNS[i]} not found in bundle — ` +
        'update STATE_LINE_PATTERNS to match the new bundle shape'
    )
  }

  assert.notEqual(
    callLineNumber,
    -1,
    'expected `var poolLimits = readPersistedPoolLimits()` not found in bundle'
  )

  const latestStateLine = Math.max(...stateLineNumbers)

  assert.ok(
    latestStateLine < callLineNumber,
    `rememberLog state must be declared before readPersistedPoolLimits() ` +
      `is called. Latest state line: ${latestStateLine}, call line: ${callLineNumber}. ` +
      'See issue #102018 and commit 05f548f35d — moving the call (or any of ' +
      'the four `var` declarations) without re-checking this assertion will ' +
      're-introduce the "Cannot read properties of undefined (reading push)" ' +
      'crash at desktop launch.'
  )
})

test('anti-regression: bundle with state declared AFTER the call must fail the ordering assertion', async () => {
  // Sanity-check that the test above isn't tautological. We synthesize a
  // bundle-shaped string that intentionally places `var poolLimits =
  // readPersistedPoolLimits()` BEFORE the state declarations — exactly the
  // pre-fix main.ts shape — and confirm the same ordering predicate flags it.
  const synth = [
    'var poolLimits = readPersistedPoolLimits();', // line 0
    'var hermesLog = [];', // line 1 — TOO LATE
    'var desktopLogBuffer = "";', // line 2
    'var desktopLogFlushTimer = null;', // line 3
    'var desktopLogFlushPromise = Promise.resolve();' // line 4
  ].join('\n')

  const synthLines = synth.split('\n')

  const synthStateLines = STATE_LINE_PATTERNS.map((re) =>
    synthLines.findIndex((l) => re.test(l.trim()))
  )

  const synthCallLine = synthLines.findIndex((l) =>
    CALL_LINE_PATTERN.test(l.trim())
  )

  const synthLatestState = Math.max(...synthStateLines)

  assert.ok(
    synthLatestState > synthCallLine,
    'synthetic bundle should have state AFTER the call (this is the bug shape); ' +
      'if you are reading this, the test setup is broken.'
  )

  // The same predicate the real test uses must reject this layout.
  assert.ok(
    !(synthLatestState < synthCallLine),
    'synthetic reverse-order bundle was incorrectly accepted as ordered'
  )
})

// `readFile` is imported only to keep the import graph honest — without
// it, future refactors that delete the import would also silently delete
// the file this test is trying to assert against. Touching it on every
// run makes the failure mode (missing main.ts) loud.
test('main.ts exists at the expected path', async () => {
  const src = await readFile(MAIN_ENTRY, 'utf8')
  assert.ok(
    src.includes('readPersistedPoolLimits'),
    'main.ts no longer references readPersistedPoolLimits; this test guards ' +
      'against the function being renamed/removed without an update here.'
  )
})
