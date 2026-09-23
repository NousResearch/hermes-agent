import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { resolve } from 'node:path'

import { test } from 'vitest'

// hud-modifier-gesture.h is the clean-tap state machine the macOS (.m) and
// Linux XI2 (-x11.c) helpers both include. Compile its C contract and run it
// wherever a C toolchain ships with the runner: Linux (the JS CI lane) and
// macOS. Windows uses the separate C# port.
test.skipIf(process.platform === 'win32')(
  'native modifier gesture state machine only summons on a clean, bounded two-modifier tap',
  () => {
    const dir = mkdtempSync(resolve(tmpdir(), 'hermes-hud-gesture-'))

    try {
      const binary = resolve(dir, 'gesture')
      const source = resolve(import.meta.dirname, 'native', 'hud-modifier-gesture.test.c')
      const flags = ['-std=c11', '-Wall', '-Wextra', '-Werror', source, '-o', binary]

      if (process.platform === 'darwin') {
        execFileSync('xcrun', ['--sdk', 'macosx', 'clang', ...flags])
      } else {
        execFileSync('cc', flags)
      }

      assert.match(execFileSync(binary, [], { encoding: 'utf8', timeout: 5_000 }), /assertions passed/)
    } finally {
      rmSync(dir, { force: true, recursive: true })
    }
  },
  30_000
)
