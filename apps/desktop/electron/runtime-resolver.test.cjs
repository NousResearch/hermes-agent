'use strict'

const assert = require('node:assert/strict')
const test = require('node:test')

const { hasUsableActiveInstall } = require('./runtime-resolver.cjs')

function baseOpts(overrides = {}) {
  return {
    activeRoot: '/home/user/.hermes/hermes-agent',
    canImportHermesCli: () => true,
    fileExists: () => true,
    getVenvPython: venvRoot => `${venvRoot}/bin/python`,
    isHermesSourceRoot: () => true,
    rememberLog: () => {},
    venvRoot: '/home/user/.hermes/hermes-agent/venv',
    ...overrides
  }
}

test('hasUsableActiveInstall accepts a CLI-first source checkout with a working venv', () => {
  assert.equal(hasUsableActiveInstall(baseOpts()), true)
})

test('hasUsableActiveInstall rejects missing source roots and venvs', () => {
  assert.equal(hasUsableActiveInstall(baseOpts({ isHermesSourceRoot: () => false })), false)
  assert.equal(hasUsableActiveInstall(baseOpts({ fileExists: () => false })), false)
})

test('hasUsableActiveInstall rejects venvs that cannot import hermes_cli and logs why', () => {
  const lines = []
  const ok = hasUsableActiveInstall(
    baseOpts({
      canImportHermesCli: () => false,
      rememberLog: line => lines.push(line)
    })
  )

  assert.equal(ok, false)
  assert.match(lines.join('\n'), /hermes_cli is not importable/)
})
