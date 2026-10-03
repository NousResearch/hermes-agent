import { describe, expect, it } from 'vitest'

import { deferred } from '../test/deferred'

import { $settingsLock, bindSettingsLockBackend, syncSettingsLock } from './settings-lock'

const LOCKED = { enabled: true, keys: ['approvals.mode'], unlocked: false }

describe('settings-lock status belongs to the backend and moment it was read for', () => {
  it.each(['answers', 'fails'] as const)(
    'a slow read from the previous backend that %s late never replaces the current one',
    async outcome => {
      const slow = deferred<unknown>()
      const backendA = () => slow.promise
      const backendB = async () => LOCKED

      bindSettingsLockBackend(backendA)
      const lateA = syncSettingsLock(backendA)

      bindSettingsLockBackend(backendB)
      await syncSettingsLock(backendB)
      expect($settingsLock.get().enabled).toBe(true)

      if (outcome === 'answers') {
        slow.resolve({ enabled: false })
      } else {
        slow.reject(new Error('backend A went away'))
      }

      await lateA
      expect($settingsLock.get()).toMatchObject({ enabled: true, keys: ['approvals.mode'], unlocked: false })
    }
  )
})
