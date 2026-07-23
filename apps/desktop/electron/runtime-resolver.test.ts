import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import { hasUsableActiveInstall } from './runtime-resolver'

function baseOptions(overrides = {}) {
  return {
    activeRoot: '/home/user/.hermes/hermes-agent',
    venvRoot: '/home/user/.hermes/hermes-agent/venv',
    getVenvPython: (venvRoot: string) => `${venvRoot}/bin/python`,
    isHermesSourceRoot: () => true,
    fileExists: () => true,
    canImportHermesCli: () => true,
    existingPythonPath: '/existing/modules',
    pathDelimiter: path.delimiter,
    ...overrides
  }
}

test('accepts an unmarked source runtime after a fresh health probe', () => {
  let probePython = ''
  let probeEnv: Record<string, string> | undefined

  const usable = hasUsableActiveInstall(
    baseOptions({
      canImportHermesCli: (python: string, opts: { env?: Record<string, string> }) => {
        probePython = python
        probeEnv = opts.env
        return true
      }
    })
  )

  assert.equal(usable, true)
  assert.equal(probePython, '/home/user/.hermes/hermes-agent/venv/bin/python')
  assert.equal(
    probeEnv?.PYTHONPATH,
    ['/home/user/.hermes/hermes-agent', '/existing/modules'].join(path.delimiter)
  )
})

test('never caches a positive probe across runtime checks', () => {
  let probeCount = 0
  const options = baseOptions({
    canImportHermesCli: () => {
      probeCount += 1
      return probeCount === 1
    }
  })

  assert.equal(hasUsableActiveInstall(options), true)
  assert.equal(hasUsableActiveInstall(options), false)
  assert.equal(probeCount, 2)
})

test('rejects missing source roots, interpreters, and failed imports', () => {
  assert.equal(hasUsableActiveInstall(baseOptions({ isHermesSourceRoot: () => false })), false)
  assert.equal(hasUsableActiveInstall(baseOptions({ fileExists: () => false })), false)
  assert.equal(hasUsableActiveInstall(baseOptions({ canImportHermesCli: () => false })), false)
})
