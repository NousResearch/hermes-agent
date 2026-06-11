/**
 * Tests for desktop active install detection.
 *
 * Run with: node --test apps/desktop/electron/active-install.test.cjs
 */

const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const { getVenvPython, hasUsableActiveInstall } = require('./active-install.cjs')

function makeTempInstall() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-active-install-'))
  const activeRoot = path.join(root, 'hermes-agent')
  const venvRoot = path.join(activeRoot, 'venv')
  fs.mkdirSync(path.join(activeRoot, 'hermes_cli'), { recursive: true })
  fs.writeFileSync(path.join(activeRoot, 'hermes_cli', 'main.py'), '# hermes cli entrypoint\n')
  fs.mkdirSync(path.dirname(getVenvPython(venvRoot, 'darwin')), { recursive: true })
  fs.writeFileSync(getVenvPython(venvRoot, 'darwin'), '#!/usr/bin/env python3\n')
  return { root, activeRoot, venvRoot }
}

test('valid canonical install is usable even when bootstrap marker is missing', () => {
  const { root, activeRoot, venvRoot } = makeTempInstall()
  try {
    assert.equal(hasUsableActiveInstall(activeRoot, venvRoot, { platform: 'darwin' }), true)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('active install readiness still requires a venv python', () => {
  const { root, activeRoot, venvRoot } = makeTempInstall()
  try {
    fs.rmSync(getVenvPython(venvRoot, 'darwin'), { force: true })
    assert.equal(hasUsableActiveInstall(activeRoot, venvRoot, { platform: 'darwin' }), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('active install readiness still requires hermes_cli source root', () => {
  const { root, activeRoot, venvRoot } = makeTempInstall()
  try {
    fs.rmSync(path.join(activeRoot, 'hermes_cli'), { recursive: true, force: true })
    assert.equal(hasUsableActiveInstall(activeRoot, venvRoot, { platform: 'darwin' }), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
