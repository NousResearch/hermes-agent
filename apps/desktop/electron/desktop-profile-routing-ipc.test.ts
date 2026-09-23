import assert from 'node:assert/strict'

import { test } from 'vitest'

import { registerDesktopProfileRoutingIpc } from './desktop-profile-routing-ipc'

test('profile IPC preserves registration order, persistence, guarded rehome, and live window reload', async () => {
  const handlers = new Map<string, (...args: any[]) => Promise<any>>()
  const order: string[] = []
  const effects: string[] = []
  let active = 'default'

  registerDesktopProfileRoutingIpc({
    ipcMain: {
      handle: (channel: string, callback: (...args: any[]) => Promise<any>) => {
        order.push(channel)
        handlers.set(channel, callback)
      }
    },
    desktopProfilePreferences: {
      getDefault: () => ({ connectionId: 'local', profile: 'default' }),
      setDefault: (route: unknown) => route
    },
    readActiveDesktopProfile: () => active,
    writeActiveDesktopProfile: (name: string) => {
      effects.push(`remember:${name}`)
      active = name

      return name
    },
    assertCanMutateManagedPrimaryRouting: () => effects.push('guard'),
    teardownPrimaryBackendAndWait: async () => effects.push('teardown'),
    getMainWindow: () => ({ reload: () => effects.push('reload') })
  } as any)

  assert.deepEqual(order, [
    'hermes:profile:default:get',
    'hermes:profile:default:set',
    'hermes:profile:get',
    'hermes:profile:remember',
    'hermes:profile:set'
  ])
  assert.deepEqual(await handlers.get('hermes:profile:get')!(null), { profile: 'default' })
  assert.deepEqual(await handlers.get('hermes:profile:remember')!(null, 'work'), { profile: 'work' })
  assert.deepEqual(effects, ['remember:work'])
  assert.deepEqual(await handlers.get('hermes:profile:set')!(null, 'other'), { profile: 'other' })
  assert.deepEqual(effects, ['remember:work', 'guard', 'remember:other', 'teardown', 'reload'])
})
