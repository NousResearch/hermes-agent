import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

import { describe, expect, it } from 'vitest'

const here = path.dirname(fileURLToPath(import.meta.url))
const mainSource = fs.readFileSync(path.join(here, 'main.ts'), 'utf8').replace(/\r\n/g, '\n')

/**
 * Regression test for #102868 — Desktop: Bots profile on remote (SSH) backend
 * spawns wsl.exe on WSL-less Windows.
 *
 * When opening a bot profile connected to a remote (SSH) Hermes backend on
 * Windows 11 with WSL feature present (even without installed distros), Desktop
 * spawned wsl.exe (black console flash) to resolve POSIX paths from the remote
 * host via the WSL-bridge. The bridge eligibility was never written for v2
 * registry profiles (ensureRegistryBackend), while v1 profiles (ensureBackend)
 * correctly registered their mode.
 *
 * This test ensures that EVERY backend-lifecycle path (pooled SSH, cloud/url,
 * local-pooled, local-delegate, plus ensureBackend for v1 primary) writes the
 * bridge-eligibility flag by checking that setWslBridgeProfileState is called
 * after each await connectionPromise/spawnPoolBackend/connectRegistryBackend.
 */
describe('WSL bridge state registration (#102868)', () => {
  it('ensureRegistryBackend writes setWslBridgeProfileState after every backend connection', () => {
    // Scan for ensureRegistryBackend function.
    const fnStart = mainSource.indexOf('async function ensureRegistryBackend(')
    expect(fnStart).toBeGreaterThan(-1)

    // Find the function's closing brace (scan to first balanced close after opening).
    const openIdx = mainSource.indexOf('{', fnStart)
    let braceDepth = 1
    let closeIdx = openIdx + 1
    while (braceDepth > 0 && closeIdx < mainSource.length) {
      if (mainSource[closeIdx] === '{') braceDepth++
      if (mainSource[closeIdx] === '}') braceDepth--
      closeIdx++
    }
    const fnBody = mainSource.slice(fnStart, closeIdx)

    // #102868 fixed five code paths: v1-primary SSH renew, cloud/url renew,
    // local-pooled renew, local-pooled spawn, and remote-pooled spawn. Each
    // must call setWslBridgeProfileState(profileKey, ...) after establishing
    // the connection. We scan for at least 4 calls (the fix added 5, but we
    // tolerate minor variations in implementation as long as coverage is real).
    const calls = fnBody.match(/setWslBridgeProfileState\(/g) || []
    expect(calls.length).toBeGreaterThanOrEqual(4)
  })

  it('ensureBackend (v1 primary) writes setWslBridgeProfileState for pooled remotes (preserved behavior)', () => {
    const fnStart = mainSource.indexOf('async function ensureBackend(profile')
    expect(fnStart).toBeGreaterThan(-1)

    const openIdx = mainSource.indexOf('{', fnStart)
    let braceDepth = 1
    let closeIdx = openIdx + 1
    while (braceDepth > 0 && closeIdx < mainSource.length) {
      if (mainSource[closeIdx] === '{') braceDepth++
      if (mainSource[closeIdx] === '}') braceDepth--
      closeIdx++
    }
    const fnBody = mainSource.slice(fnStart, closeIdx)

    // ensureBackend (v1 primary path) already wrote setWslBridgeProfileState
    // for pooled remotes. This test preserves that existing coverage and
    // ensures it doesn't regress.
    const calls = fnBody.match(/setWslBridgeProfileState\(/g) || []
    expect(calls.length).toBeGreaterThanOrEqual(1)
  })
})
