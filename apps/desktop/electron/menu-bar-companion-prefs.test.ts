import assert from 'node:assert/strict'

import { test } from 'vitest'

import { DEFAULT_MENU_BAR_COMPANION_ENABLED, menuBarCompanionEnabledFromConfig } from './menu-bar-companion-prefs'

test('menuBarCompanionEnabledFromConfig defaults when missing', () => {
  assert.equal(DEFAULT_MENU_BAR_COMPANION_ENABLED, false)
  assert.equal(menuBarCompanionEnabledFromConfig(null), DEFAULT_MENU_BAR_COMPANION_ENABLED)
  assert.equal(menuBarCompanionEnabledFromConfig({}), DEFAULT_MENU_BAR_COMPANION_ENABLED)
})

test('menuBarCompanionEnabledFromConfig respects boolean enabled', () => {
  assert.equal(menuBarCompanionEnabledFromConfig({ enabled: false }), false)
  assert.equal(menuBarCompanionEnabledFromConfig({ enabled: true }), true)
})
