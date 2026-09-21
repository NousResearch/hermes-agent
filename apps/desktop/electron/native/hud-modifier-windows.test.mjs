import assert from 'node:assert/strict'
import { execFile, spawn } from 'node:child_process'
import { once } from 'node:events'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { promisify } from 'node:util'
import { test } from 'node:test'
import { fileURLToPath } from 'node:url'

import { buildHudModifierMonitor, resolveWindowsFrameworkCompiler } from '../../scripts/build-hud-modifier-monitor.mjs'

// Run on Windows, without a developer compiler on PATH, using the real Node
// pipe transport. No input is synthesized and no user desktop is controlled.
test('Windows builds and starts its HUD helper without Clang or a developer SDK', {
  skip: process.platform !== 'win32', timeout: 60_000
}, async () => {
  const distDir = mkdtempSync(join(tmpdir(), 'hud-windows-'))
  const path = process.env.PATH
  const cc = process.env.CC
  let child
  try {
    process.env.PATH = ''
    process.env.CC = 'hermes-test-no-native-compiler'
    let binary
    try {
      binary = buildHudModifierMonitor({ distDir })
    } finally {
      process.env.PATH = path
      if (cc === undefined) delete process.env.CC
      else process.env.CC = cc
    }
    assert.ok(binary, 'A Windows build must include a working HUD helper, not silently skip it')
    const check = await promisify(execFile)(binary, ['--check'], { timeout: 15_000, windowsHide: true })
    assert.equal(JSON.parse(check.stdout.trim()).type, 'ready')
    child = spawn(binary, [], { windowsHide: true, stdio: ['pipe', 'pipe', 'pipe'] })
    const exited = once(child, 'exit')
    exited.catch(() => {})
    await new Promise((resolve, reject) => {
      let output = ''
      const timer = setTimeout(() => reject(new Error('HUD helper did not become ready')), 15_000)
      child.once('error', error => { clearTimeout(timer); reject(error) })
      child.once('exit', (code, signal) => {
        clearTimeout(timer)
        reject(new Error(`HUD helper exited before readiness: ${code}/${signal}`))
      })
      child.stdout.on('data', chunk => {
        output += chunk.toString()
        if (output.includes('\n')) {
          clearTimeout(timer)
          try {
            assert.equal(JSON.parse(output.trim()).type, 'ready')
            resolve()
          } catch (error) { reject(error) }
        }
      })
    })
    child.stdin.end()
    assert.deepEqual(await exited, [0, null], 'stdin EOF must release the listener and exit')
  } finally {
    if (child?.pid && child.exitCode === null) {
      const closed = once(child, 'close')
      child.kill()
      await closed
    }
    rmSync(distDir, { recursive: true, force: true })
  }
})

test('Windows gesture rejects ordinary shortcuts and recovers for the next tap', {
  skip: process.platform !== 'win32', timeout: 30_000
}, async () => {
  const dir = mkdtempSync(join(tmpdir(), 'hud-gesture-'))
  const exe = join(dir, 'gesture.exe')
  const run = promisify(execFile)
  try {
    await run(resolveWindowsFrameworkCompiler(), [
      '/nologo', '/target:exe', '/warnaserror+', `/out:${exe}`,
      fileURLToPath(new URL('./hud-modifier-gesture.cs', import.meta.url)),
      fileURLToPath(new URL('./hud-modifier-gesture.test.cs', import.meta.url))
    ], { windowsHide: true })
    const result = await run(exe, [], { windowsHide: true })
    assert.equal(result.stdout.trim(), 'gesture contracts passed')
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
})
