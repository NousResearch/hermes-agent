/**
 * Regression test for issue #61898 - Multiple PowerShell windows appear
 * during update + race condition in update flow.
 *
 * The bug: applyUpdates() and handOffWindowsBootstrapRecovery() spawned
 * the updater with windowsHide: false, making PowerShell windows visible.
 * Additionally, applyUpdates() reset updateInFlight in a finally block
 * immediately on return, opening a 2.5s window during which a second
 * updater could be spawned (race condition).
 *
 * The fix:
 * 1. Change windowsHide: false → true in both spawn calls
 * 2. Move updateInFlight reset from `finally` to `catch` in applyUpdates
 *    so the flag stays set through the quit dwell
 * 3. Add updateInFlight guard at the start of handOffWindowsBootstrapRecovery
 *    so it can't race with applyUpdates
 *
 * These tests use static source-string matching because the actual functions
 * require a full Electron runtime.
 */
import assert from 'node:assert/strict'
import test from 'node:test'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const mainTsPath = path.resolve(__dirname, './main.ts')
const source = fs.readFileSync(mainTsPath, 'utf-8')

test('issue #61898: applyUpdates spawn uses windowsHide: true', () => {
  // Find the applyUpdates block (between "applyUpdates — hand off" and "async function handOffWindowsBootstrapRecovery")
  const applyUpdatesMatch = source.match(
    /\/\/ applyUpdates[\s\S]*?async function applyUpdates[\s\S]*?return \{ ok: true, handedOff: true, updater \}[\s\S]*?\}/
  )
  assert.ok(applyUpdatesMatch, 'applyUpdates function block not found in main.ts')

  const block = applyUpdatesMatch[0]

  // The spawn call inside must use windowsHide: true
  const spawnMatch = block.match(/spawn\(updater,[\s\S]*?windowsHide:\s*(true|false)/)
  assert.ok(spawnMatch, 'spawn call with windowsHide not found in applyUpdates')
  assert.equal(spawnMatch[1], 'true',
    'Issue #61898: applyUpdates still spawns updater with windowsHide: false (causes visible PowerShell windows)')
})

test('issue #61898: handOffWindowsBootstrapRecovery spawn uses windowsHide: true', () => {
  const recoveryMatch = source.match(
    /async function handOffWindowsBootstrapRecovery[\s\S]*?return true\n\}/
  )
  assert.ok(recoveryMatch, 'handOffWindowsBootstrapRecovery function block not found')

  const block = recoveryMatch[0]
  const spawnMatch = block.match(/spawn\(updater,[\s\S]*?windowsHide:\s*(true|false)/)
  assert.ok(spawnMatch, 'spawn call with windowsHide not found in handOffWindowsBootstrapRecovery')
  assert.equal(spawnMatch[1], 'true',
    'Issue #61898: handOffWindowsBootstrapRecovery still spawns updater with windowsHide: false')
})

test('issue #61898: applyUpdates does NOT clear updateInFlight in finally', () => {
  // The fix moves the reset from `finally` to `catch`. If there's still a
  // `} finally { ... updateInFlight = false }` block, the race is back.
  // Use a more inclusive regex that matches to the end of the function.
  const applyUpdatesMatch = source.match(
    /\/\/ applyUpdates[\s\S]*?async function applyUpdates[\s\S]*?^\}/m
  )
  assert.ok(applyUpdatesMatch, 'applyUpdates function not found')
  const block = applyUpdatesMatch[0]

  assert.ok(
    !/}\s*finally\s*\{[\s\S]*?updateInFlight\s*=\s*false/.test(block),
    'Issue #61898: applyUpdates still resets updateInFlight in a finally block (race condition)'
  )
})

test('issue #61898: handOffWindowsBootstrapRecovery guards on updateInFlight', () => {
  const recoveryMatch = source.match(
    /async function handOffWindowsBootstrapRecovery[\s\S]*?return true\n\}/
  )
  assert.ok(recoveryMatch)
  const block = recoveryMatch[0]

  // Must have an `if (updateInFlight) return false` guard at the start
  assert.ok(
    /if\s*\(\s*updateInFlight\s*\)\s*\{?\s*return\s+false/.test(block),
    'Issue #61898: handOffWindowsBootstrapRecovery is missing updateInFlight guard'
  )
})

test('issue #61898: handOffWindowsBootstrapRecovery sets updateInFlight=true', () => {
  const recoveryMatch = source.match(
    /async function handOffWindowsBootstrapRecovery[\s\S]*?return true\n\}/
  )
  assert.ok(recoveryMatch)
  const block = recoveryMatch[0]

  // Must have `updateInFlight = true` after the guard
  // (intentionally shares the flag with applyUpdates)
  assert.ok(
    /updateInFlight\s*=\s*true/.test(block),
    'Issue #61898: handOffWindowsBootstrapRecovery does not set updateInFlight = true'
  )
})

test('issue #61898: source has no remaining windowsHide: false in update paths', () => {
  // Final sanity check: the issue specifically calls out both applyUpdates
  // and handOffWindowsBootstrapRecovery. After the fix, neither should have
  // windowsHide: false in their spawn calls.
  const applyUpdatesMatch = source.match(
    /async function applyUpdates[\s\S]*?return \{ ok: true, handedOff: true, updater \}[\s\S]*?\}/
  )
  const recoveryMatch = source.match(
    /async function handOffWindowsBootstrapRecovery[\s\S]*?return true\n\}/
  )

  for (const [name, match] of [
    ['applyUpdates', applyUpdatesMatch],
    ['handOffWindowsBootstrapRecovery', recoveryMatch]
  ]) {
    assert.ok(match, `${name} block not found`)
    const block = match[0]
    assert.ok(
      !/windowsHide:\s*false/.test(block),
      `${name} still has windowsHide: false`
    )
  }
})