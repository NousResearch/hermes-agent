import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import { spawnSync } from 'node:child_process'
import path from 'node:path'
import { test } from 'vitest'
import { prepareDesktopNativeDependencies } from './stage-native-deps.mjs'

test('the native admission child refuses ordinary Node instead of certifying an Electron ABI', () => {
  const child = spawnSync(process.execPath, [path.join(import.meta.dirname, 'probe-prepared-native.mjs'), '--child', import.meta.dirname], {
    encoding: 'utf8', timeout: 5000, env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
  })
  assert.notEqual(child.status, 0)
  assert.match(child.stderr, /Electron runtime required/)
})

test.runIf(process.platform === 'win32')('the Electron PTY probe exits after its shell despite ConPTY workers', async () => {
  const source = path.resolve(import.meta.dirname, '../../..')
  const out = fs.mkdtempSync(path.join(os.tmpdir(), 'native-exit-'))
  try {
    const native = path.join(out, 'native')
    prepareDesktopNativeDependencies({ source, out: native })
    const { default: electron } = await import('electron')
    const child = spawnSync(electron, [path.join(import.meta.dirname, 'probe-prepared-native.mjs'), '--child', native], {
      cwd: out, encoding: 'utf8', timeout: 15000, env: { ...process.env, ELECTRON_RUN_AS_NODE: '1' },
    })
    assert.equal(child.error, undefined)
    assert.equal(child.status, 0, child.stderr)
    const result = JSON.parse(child.stdout)
    assert.equal(result.arch, process.arch)
    assert.equal(result.exitCode, 0)
    assert.match(result.marker, /^hermes-pty-/)
  } finally {
    fs.rmSync(out, { recursive: true, force: true })
  }
}, 30000)
