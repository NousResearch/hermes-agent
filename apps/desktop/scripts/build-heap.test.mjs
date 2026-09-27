import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

import { test } from 'vitest'

const require = createRequire(import.meta.url)

// The vite/rolldown transform of the desktop module tree needs more than the
// default V8 old-space limit (~4 GiB even on large machines), so the build
// step must raise its own heap instead of inheriting the default — exactly
// like the sibling `builder` script. Without this the source build dies with
// "FATAL ERROR: Zone Allocation failed - process out of memory" regardless of
// free system memory (#125502).
test('the build script raises the V8 old-space ceiling for the transform', () => {
  const build = require('../package.json').scripts.build
  // cross-env keeps the flag portable across Windows and Unix script shells.
  assert.match(build, /cross-env NODE_OPTIONS=--max-old-space-size=\d+/)
  const mb = Number(/--max-old-space-size=(\d+)/.exec(build)?.[1])
  // Above the 4 GiB default cap where the OOM was observed; in family with the
  // TUI entry (8192) rather than the packaging-only 16384.
  assert.ok(mb >= 6144 && mb <= 16384, `unexpected heap ceiling: ${mb}`)
})
