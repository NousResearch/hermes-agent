import assert from 'node:assert/strict'

import { test } from 'vitest'

import { installApplicationMenuAfterFirstWindow } from './application-menu-startup'

// #115332: a key equivalent routed through the application menu's delegate
// before any window exists segfaults the macOS shell on the updater relaunch.
function record() {
  const order: string[] = []

  installApplicationMenuAfterFirstWindow<string>({
    buildMenu: () => {
      order.push('build-menu')

      return 'menu'
    },
    setApplicationMenu: menu => order.push(`set-menu:${menu}`),
    createWindow: () => order.push('create-window')
  })

  return order
}

test('macOS installs the application menu only after the first window exists', () => {
  assert.deepEqual(record(), ['create-window', 'build-menu', 'set-menu:menu'])
})

test('Windows and Linux install the menu after the first window for native roles', () => {
  assert.deepEqual(record(), ['create-window', 'build-menu', 'set-menu:menu'])
})
