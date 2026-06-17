'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const { normalizeDesktopAppTarget } = require('./desktop-app-launch.cjs')

test('normalizeDesktopAppTarget resolves allowed Chinese app names', () => {
  assert.deepEqual(normalizeDesktopAppTarget('记事本'), {
    id: 'notepad',
    label: '记事本',
    target: 'notepad.exe',
    aliases: ['notepad', '记事本', '文本编辑器']
  })
})

test('normalizeDesktopAppTarget resolves common browser aliases', () => {
  assert.equal(normalizeDesktopAppTarget('chrome浏览器')?.id, 'chrome')
  assert.equal(normalizeDesktopAppTarget('Microsoft Edge')?.id, 'edge')
})

test('normalizeDesktopAppTarget rejects unknown apps', () => {
  assert.equal(normalizeDesktopAppTarget('powershell'), null)
  assert.equal(normalizeDesktopAppTarget('cmd'), null)
})
