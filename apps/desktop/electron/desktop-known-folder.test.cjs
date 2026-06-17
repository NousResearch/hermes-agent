'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const { normalizeDesktopFolderTarget } = require('./desktop-known-folder.cjs')

test('normalizeDesktopFolderTarget resolves common Chinese folder aliases', () => {
  assert.equal(normalizeDesktopFolderTarget('桌面')?.id, 'desktop')
  assert.equal(normalizeDesktopFolderTarget('下载')?.id, 'downloads')
  assert.equal(normalizeDesktopFolderTarget('文档')?.id, 'documents')
})

test('normalizeDesktopFolderTarget resolves English aliases', () => {
  assert.equal(normalizeDesktopFolderTarget('desktop')?.id, 'desktop')
  assert.equal(normalizeDesktopFolderTarget('downloads')?.id, 'downloads')
})

test('normalizeDesktopFolderTarget rejects unknown folders', () => {
  assert.equal(normalizeDesktopFolderTarget('system32'), null)
})
