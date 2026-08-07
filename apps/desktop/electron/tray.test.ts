import assert from 'node:assert/strict'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, beforeEach, describe, it } from 'vitest'

import {
  buildTrayMenuItems,
  readPersistedMinimizeToTray,
  writePersistedMinimizeToTray
} from './tray'

let dir: string

beforeEach(() => {
  dir = mkdtempSync(join(tmpdir(), 'hermes-tray-test-'))
})

afterEach(() => {
  rmSync(dir, { recursive: true, force: true })
})

describe('readPersistedMinimizeToTray', () => {
  it('defaults to false when nothing is persisted', () => {
    assert.equal(readPersistedMinimizeToTray(dir), false)
  })

  it('reads the persisted enabled flag', () => {
    writePersistedMinimizeToTray(dir, true)
    assert.equal(readPersistedMinimizeToTray(dir), true)
  })

  it('falls back to false on malformed JSON', () => {
    writeFileSync(join(dir, 'minimize-to-tray.json'), '{ not valid json', 'utf8')
    assert.equal(readPersistedMinimizeToTray(dir), false)
  })

  it('falls back to false when enabled is not a boolean', () => {
    writeFileSync(join(dir, 'minimize-to-tray.json'), JSON.stringify({ enabled: 'yes' }), 'utf8')
    assert.equal(readPersistedMinimizeToTray(dir), false)
  })
})

describe('writePersistedMinimizeToTray', () => {
  it('round-trips through the persisted file', () => {
    writePersistedMinimizeToTray(dir, true)
    assert.equal(readPersistedMinimizeToTray(dir), true)

    writePersistedMinimizeToTray(dir, false)
    assert.equal(readPersistedMinimizeToTray(dir), false)
  })

  it('creates the userData directory if missing', () => {
    const nested = join(dir, 'deep', 'nested')
    writePersistedMinimizeToTray(nested, true)
    assert.equal(readPersistedMinimizeToTray(nested), true)
  })
})

describe('buildTrayMenuItems', () => {
  const deps = (overrides: Partial<Parameters<typeof buildTrayMenuItems>[0]> = {}) =>
    buildTrayMenuItems({
      isWindowVisible: false,
      locale: 'en',
      onToggleVisibility: () => {},
      onNewSession: () => {},
      onOpenSettings: () => {},
      onQuit: () => {},
      ...overrides
    })

  it('shows "Show Window" when the window is hidden (en)', () => {
    const items = deps({ isWindowVisible: false })
    assert.equal(items[0].label, 'Show Window')
  })

  it('shows "Hide Window" when the window is visible (en)', () => {
    const items = deps({ isWindowVisible: true })
    assert.equal(items[0].label, 'Hide Window')
  })

  it('localizes the menu to the renderer locale', () => {
    const items = deps({ locale: 'zh' })
    const labels = items.filter(i => 'label' in i).map(i => (i as { label: string }).label)
    assert.deepEqual(labels, ['显示窗口', '新建会话', '打开设置', '退出'])
  })

  it('falls back to English for an unknown locale', () => {
    const items = deps({ locale: 'xx' })
    assert.equal(items[0].label, 'Show Window')
    assert.equal(items[4].label, 'Quit')
  })

  it('exposes new-session / open-settings / quit entries in order', () => {
    const items = deps()
    const labels = items.filter(i => 'label' in i).map(i => (i as { label: string }).label)
    assert.deepEqual(labels, ['Show Window', 'New Session', 'Open Settings', 'Quit'])
  })

  it('omits a separator from the label list but keeps the structure', () => {
    const items = deps()
    // separator is the 4th entry (index 3)
    assert.equal((items[3] as { type?: string }).type, 'separator')
  })

  it('wires the quit callback', () => {
    let quitCalled = false

    const items = deps({ onQuit: () => { quitCalled = true } })

    ;(items[4] as { click: () => void }).click()
    assert.equal(quitCalled, true)
  })
})
