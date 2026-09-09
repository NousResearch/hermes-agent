import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  APPLICATION_MENU_LOCALES,
  type ApplicationMenuActions,
  buildApplicationMenuTemplate,
  isApplicationMenuLocale
} from './application-menu'

const actions: ApplicationMenuActions = {
  checkForUpdates: () => undefined,
  close: () => undefined,
  createWindow: () => undefined,
  openFolder: () => undefined,
  reloadPreview: () => undefined,
  resetZoom: () => undefined,
  showAbout: () => undefined,
  toggleDevTools: () => undefined,
  zoomIn: () => undefined,
  zoomOut: () => undefined
}

function menuShape(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(menuShape)
  }

  if (!value || typeof value !== 'object') {
    return value
  }

  const item = value as Record<string, unknown>

  return {
    accelerator: item.accelerator,
    role: item.role,
    submenu: menuShape(item.submenu),
    type: item.type
  }
}

describe('application menu locale validation', () => {
  test('accepts only canonical Hermes display locales', () => {
    for (const locale of ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru']) {
      assert.equal(isApplicationMenuLocale(locale), true)
    }

    for (const locale of ['zh-TW', 'zh_Hant', 'en-US', 'de', '', null, 1, {}]) {
      assert.equal(isApplicationMenuLocale(locale), false)
    }
  })
})

describe('buildApplicationMenuTemplate', () => {
  test('localizes every visible macOS item in Traditional Chinese', () => {
    const template = buildApplicationMenuTemplate({ actions, appName: 'Hermes', isMac: true, locale: 'zh-hant' })

    assert.deepEqual(
      template.map(item => item.label),
      ['Hermes', '檔案', '編輯', '顯示方式', '視窗', '說明']
    )

    const appMenu = template[0].submenu as Electron.MenuItemConstructorOptions[]
    const fileMenu = template[1].submenu as Electron.MenuItemConstructorOptions[]
    const editMenu = template[2].submenu as Electron.MenuItemConstructorOptions[]
    const viewMenu = template[3].submenu as Electron.MenuItemConstructorOptions[]
    const windowMenu = template[4].submenu as Electron.MenuItemConstructorOptions[]
    const helpMenu = template[5].submenu as Electron.MenuItemConstructorOptions[]

    assert.deepEqual(appMenu.map(item => item.label), [
      '關於 Hermes',
      '檢查更新…',
      undefined,
      '服務',
      undefined,
      '隱藏 Hermes',
      '隱藏其他',
      '全部顯示',
      undefined,
      '結束 Hermes'
    ])
    assert.deepEqual(fileMenu.map(item => item.label), ['新增視窗', '開啟資料夾…', undefined, '關閉'])
    assert.deepEqual(editMenu.map(item => item.label), [
      '還原',
      '重做',
      undefined,
      '剪下',
      '拷貝',
      '貼上',
      '貼上並符合樣式',
      '刪除',
      '全選'
    ])
    assert.deepEqual(viewMenu.map(item => item.label), [
      '重新載入',
      '強制重新載入',
      '切換開發者工具',
      undefined,
      '實際大小',
      '放大',
      '縮小',
      undefined,
      '切換全螢幕'
    ])
    assert.deepEqual(windowMenu.map(item => item.label), ['最小化', '縮放', '全部移到最前面'])
    assert.deepEqual(helpMenu.map(item => item.label), ['檢查更新…'])
  })

  test('changes labels without changing roles, accelerators, separators, or callbacks', () => {
    const english = buildApplicationMenuTemplate({ actions, appName: 'Hermes', isMac: true, locale: 'en' })
    const traditional = buildApplicationMenuTemplate({ actions, appName: 'Hermes', isMac: true, locale: 'zh-hant' })

    assert.deepEqual(menuShape(traditional), menuShape(english))
    assert.equal(english[1].label, 'File')
    assert.equal(traditional[1].label, '檔案')
  })

  test('provides non-empty localized labels for every current desktop locale, including Arabic', () => {
    const english = buildApplicationMenuTemplate({ actions, appName: 'Hermes', isMac: true, locale: 'en' })

    for (const locale of APPLICATION_MENU_LOCALES) {
      const template = buildApplicationMenuTemplate({ actions, appName: 'Hermes', isMac: true, locale })
      const items = template.flatMap(item => [item, ...((item.submenu || []) as Electron.MenuItemConstructorOptions[])])

      for (const item of items) {
        if (item.type !== 'separator') {
          assert.ok(item.label?.trim(), `${locale} has an unlabeled visible menu item`)
        }
      }

      if (locale !== 'en') {
        assert.notEqual(template[1].label, english[1].label, `${locale} silently fell back to English`)
      }
    }
  })

  test('keeps the previous non-mac menu shape', () => {
    const template = buildApplicationMenuTemplate({ actions, appName: 'Hermes', isMac: false, locale: 'en' })

    assert.deepEqual(template.map(item => item.label), ['File', 'Edit', 'View', 'Window', 'Help'])
    assert.equal((template[0].submenu as Electron.MenuItemConstructorOptions[])[3].role, 'quit')
  })
})
