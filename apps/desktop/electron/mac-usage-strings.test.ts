// TCC usage strings live in electron-builder.config.cjs's mac.extendInfo —
// without them a hardened-runtime (signed) build is denied Contacts / Apple
// Events access silently: macOS requires the usage description BEFORE it will
// even show the permission prompt (#59482). These tests hold that contract at
// the packaging seam so the strings can't silently disappear in a config move
// again (they were lost once already when the builder config moved out of
// package.json).
import assert from 'node:assert/strict'
import { createRequire } from 'node:module'

import { test } from 'vitest'

const require: NodeJS.Require = createRequire(import.meta.url)

const MAC_USAGE_STRINGS: readonly [string, string][] = [
  ['NSContactsUsageDescription', 'Contacts'],
  ['NSAppleEventsUsageDescription', 'Apple Events']
]

test('mac extendInfo declares a usage string for every TCC-gated desktop service', () => {
  const config = require('../electron-builder.config.cjs')
  const extendInfo = config?.mac?.extendInfo

  assert.ok(extendInfo, 'electron-builder.config.cjs must define mac.extendInfo')

  for (const [key, service] of MAC_USAGE_STRINGS) {
    const value = extendInfo[key]

    assert.ok(
      typeof value === 'string' && value.trim().length > 0,
      `mac.extendInfo.${key} must be a non-empty usage string; without it macOS denies ${service} access without showing a TCC prompt (#59482)`
    )
  }
})
