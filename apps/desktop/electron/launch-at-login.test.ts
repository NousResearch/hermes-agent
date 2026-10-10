import assert from 'node:assert/strict'

import { test } from 'vitest'

import { launchAtLoginSettingsFor } from './launch-at-login'

test('a packaged install registers the exact executable that is running', () => {
  assert.deepEqual(launchAtLoginSettingsFor(true, 'C:\\Users\\me\\AppData\\Local\\Hermes\\Hermes.exe'), {
    enabled: true,
    path: 'C:\\Users\\me\\AppData\\Local\\Hermes\\Hermes.exe'
  })
})

test('without a path Electron falls back to the running binary', () => {
  assert.deepEqual(launchAtLoginSettingsFor(true), { enabled: true })
  assert.deepEqual(launchAtLoginSettingsFor(true, null), { enabled: true })
  assert.deepEqual(launchAtLoginSettingsFor(true, ''), { enabled: true })
})

test('disabling is a plain flag — never a path rewrite', () => {
  assert.deepEqual(launchAtLoginSettingsFor(false, '/usr/bin/hermes'), { enabled: false, path: '/usr/bin/hermes' })
  assert.deepEqual(launchAtLoginSettingsFor(false), { enabled: false })
})

test('anything but a real true registers nothing', () => {
  // The caller can hand back a parsed JSON value; a truthy string must not
  // silently write a login item the user never asked for.
  assert.equal(launchAtLoginSettingsFor('yes' as unknown as boolean).enabled, false)
})
