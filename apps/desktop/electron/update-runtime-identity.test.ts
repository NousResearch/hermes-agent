import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { classifyDesktopUpdateRuntime, probeDesktopUpdateRuntime } from './update-runtime-identity'

test('split-brain runtime refuses the git updater when the selected venv imports an immutable generation', () => {
  const checkout = path.resolve('/Users/hermes/.hermes/hermes-agent')

  const generation = path.resolve(
    '/Users/hermes/.hermes/releases/nimble-reset5/generations/abc/source/repository'
  )

  assert.deepEqual(classifyDesktopUpdateRuntime({ updateRoot: checkout, runtimeRoot: generation }), {
    kind: 'managed-runtime',
    updateRoot: checkout,
    runtimeRoot: generation,
    supported: false,
    message:
      'This Hermes runtime is managed by an immutable release. Update it through its release manager; Desktop will not run the Git updater against a different checkout.'
  })
})

test('matching mutable checkout keeps the desktop git updater enabled', () => {
  const checkout = path.resolve('/Users/hermes/.hermes/hermes-agent')

  assert.deepEqual(classifyDesktopUpdateRuntime({ updateRoot: checkout, runtimeRoot: checkout }), {
    kind: 'git-checkout',
    updateRoot: checkout,
    runtimeRoot: checkout,
    supported: true,
    message: null
  })
})

test('runtime probe asks the selected venv interpreter for the effective Hermes import root', () => {
  const checkout = path.resolve('/Users/hermes/.hermes/hermes-agent')
  const generation = path.resolve('/Users/hermes/.hermes/releases/nimble-reset5/generations/abc/source/repository')
  const calls: Array<{ command: string; args: string[] }> = []

  const result = probeDesktopUpdateRuntime(checkout, {
    execFileSync: (command, args) => {
      calls.push({ command, args })

      return `${generation}\n`
    },
    isWindows: false
  })

  assert.equal(result.runtimeRoot, generation)
  assert.equal(result.kind, 'managed-runtime')
  assert.equal(result.supported, false)
  assert.equal(calls.length, 1)
  assert.equal(calls[0].command, path.join(checkout, 'venv', 'bin', 'python'))
  assert.equal(calls[0].args[0], '-c')
  assert.match(calls[0].args[1], /hermes_cli/)
})

test('failed runtime identity probe fails closed instead of invoking a possibly unrelated git updater', () => {
  const checkout = path.resolve('/Users/hermes/.hermes/hermes-agent')

  const result = probeDesktopUpdateRuntime(checkout, {
    execFileSync: () => {
      throw new Error('probe failed')
    },
    isWindows: false
  })

  assert.equal(result.kind, 'unknown-runtime')
  assert.equal(result.supported, false)
  assert.match(result.message || '', /could not verify/i)
})
