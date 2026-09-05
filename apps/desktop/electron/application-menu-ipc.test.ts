import assert from 'node:assert/strict'

import { beforeEach, describe, test, vi } from 'vitest'

const handlers = new Map<string, (...args: unknown[]) => unknown>()

vi.mock('electron', () => ({
  ipcMain: {
    handle: (channel: string, handler: (...args: unknown[]) => unknown) => handlers.set(channel, handler)
  }
}))

const { registerApplicationMenuIpc } = await import('./application-menu-ipc')

function invoke(channel: string, ...args: unknown[]) {
  const handler = handlers.get(channel)

  assert.ok(handler, `handler registered for ${channel}`)

  return handler({}, ...args)
}

describe('application menu IPC', () => {
  beforeEach(() => {
    handlers.clear()
  })

  test('forwards every canonical renderer locale', async () => {
    const setLocale = vi.fn()

    registerApplicationMenuIpc(setLocale)

    for (const locale of ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru']) {
      assert.deepEqual(await invoke('hermes:application-menu:set-locale', locale), { locale, ok: true })
    }

    assert.deepEqual(
      setLocale.mock.calls.map(call => call[0]),
      ['en', 'zh', 'zh-hant', 'ja', 'ar', 'ru']
    )
  })

  test('rejects aliases, unsupported locales, and non-strings without changing the menu', async () => {
    const setLocale = vi.fn()

    registerApplicationMenuIpc(setLocale)

    for (const locale of ['zh-TW', 'en-US', 'de', '', null, 42, {}]) {
      await assert.rejects(
        Promise.resolve().then(() => invoke('hermes:application-menu:set-locale', locale)),
        /Invalid application menu locale/
      )
    }

    assert.equal(setLocale.mock.calls.length, 0)
  })
})
