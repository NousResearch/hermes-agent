import assert from 'node:assert/strict'

import { test } from 'vitest'

import { windowMenuTemplate } from './window-menu'

test('macOS registers the native Window menu so AppKit window commands remain available', () => {
  const menu = windowMenuTemplate(true)

  assert.equal(menu.role, 'windowMenu')
  assert.deepEqual(
    menu.submenu.map(item => item.role),
    ['minimize', 'zoom', 'front']
  )
})

test('non-macOS keeps the explicit cross-platform Window menu', () => {
  assert.deepEqual(windowMenuTemplate(false), {
    label: 'Window',
    submenu: [{ role: 'minimize' }, { role: 'close' }]
  })
})
