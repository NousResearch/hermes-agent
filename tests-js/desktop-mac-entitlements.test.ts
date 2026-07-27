/**
 * Regression for #37718: macOS microphone entitlement must be inherited.
 *
 * Hermes Desktop signs with ``hardenedRuntime: true`` and points
 * electron-builder at two entitlement files (see ``apps/desktop/package.json``):
 *
 * - ``entitlements`` → ``electron/entitlements.mac.plist`` (the main app), and
 * - ``entitlementsInherit`` → ``electron/entitlements.mac.inherit.plist``
 *   (the Electron Helper / Setup processes).
 *
 * Under the hardened runtime, the process that actually opens the microphone
 * is a Helper, which inherits the *inherit* plist.
 * ``com.apple.security.device.audio-input`` lived only in the main plist, so
 * macOS' TCC layer refused the microphone with:
 *
 *     Prompting policy for hardened runtime; service: kTCCServiceMicrophone
 *     requires entitlement com.apple.security.device.audio-input but it is missing
 *
 * and never showed the permission prompt. These tests pin that every device
 * entitlement granted to the main app is also granted to the inherited helpers.
 *
 * (Ported from tests/test_desktop_mac_entitlements.py — plist assertions about
 * apps/desktop/electron/*.plist belong in the JS lane because the CI change
 * classifier skips the Python suite on apps/-only PRs.)
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import * as plist from 'plist'
import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const ELECTRON_DIR = path.join(REPO_ROOT, 'apps', 'desktop', 'electron')
const MAIN_PLIST = path.join(ELECTRON_DIR, 'entitlements.mac.plist')
const INHERIT_PLIST = path.join(ELECTRON_DIR, 'entitlements.mac.inherit.plist')

const DEVICE_PREFIX = 'com.apple.security.device.'

function loadEntitlements(plistPath: string): Record<string, unknown> {
  assert.ok(fs.existsSync(plistPath), `missing entitlements file: ${plistPath}`)
  const data = plist.parse(fs.readFileSync(plistPath, 'utf-8'))
  assert.ok(
    typeof data === 'object' && data !== null && !Array.isArray(data),
    `${path.basename(plistPath)} should parse to a dict`
  )

  return data as Record<string, unknown>
}

test('inherit plist grants microphone (regression #37718)', () => {
  const inherit = loadEntitlements(INHERIT_PLIST)
  assert.equal(
    inherit['com.apple.security.device.audio-input'],
    true,
    'entitlements.mac.inherit.plist must grant ' +
      '`com.apple.security.device.audio-input`; without it the ' +
      'hardened-runtime Helper process is denied the microphone and no ' +
      'TCC prompt appears (#37718).'
  )
})

test('every device.* entitlement on the main app is also inherited', () => {
  const main = loadEntitlements(MAIN_PLIST)
  const inherit = loadEntitlements(INHERIT_PLIST)

  const missing = Object.entries(main)
    .filter(([key, val]) => key.startsWith(DEVICE_PREFIX) && val === true)
    .map(([key]) => key)
    .filter((key) => inherit[key] !== true)

  assert.deepEqual(
    missing,
    [],
    'Device entitlements present in entitlements.mac.plist but missing from ' +
      `entitlements.mac.inherit.plist: ${JSON.stringify(missing)}. ` +
      'Helper/Setup processes inherit the latter under hardenedRuntime, so ' +
      'any device access the app needs must be listed in both (#37718).'
  )
})

for (const plist of [MAIN_PLIST, INHERIT_PLIST]) {
  test(`${path.basename(plist)} is a well-formed non-empty entitlement dict`, () => {
    const data = loadEntitlements(plist)
    assert.ok(
      Object.keys(data).length > 0,
      `${path.basename(plist)} should be a non-empty dict`
    )
  })
}
