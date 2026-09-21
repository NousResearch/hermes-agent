import assert from 'node:assert/strict'
import { test } from 'vitest'

import { ensureRolldownBinding } from './ensure-rolldown-binding.mjs'

// Shape of Node's uncaught-error output when rolldown's loader finds neither
// the native package nor the wasm fallback (cause chain, native first).
const loaderStderr = `Error: Cannot find native binding. npm has a bug related to optional dependencies (https://github.com/npm/cli/issues/4828).
  [cause]: Error: Cannot find module '@rolldown/binding-win32-x64-msvc'
    [cause]: Error: Cannot find module '@rolldown/binding-wasm32-wasi'`

test('does nothing when Rolldown already loads', () => {
  let installs = 0
  const ok = ensureRolldownBinding({
    root: '/repo',
    probe: () => ({ status: 0 }),
    install: () => {
      installs += 1
      return { status: 0 }
    }
  })

  assert.equal(ok, true)
  assert.equal(installs, 0)
})

test('installs the exact native binding the loader asked for, then re-verifies', () => {
  let probes = 0
  const installs = []
  const ok = ensureRolldownBinding({
    root: '/repo',
    probe: () => ({ status: probes++ === 0 ? 1 : 0, stderr: loaderStderr }),
    install: (_root, spec) => {
      installs.push(spec)
      return { status: 0 }
    },
    rolldownVersion: () => '1.2.1'
  })

  assert.equal(ok, true)
  assert.deepEqual(installs, ['@rolldown/binding-win32-x64-msvc@1.2.1'])
  assert.equal(probes, 2)
})
