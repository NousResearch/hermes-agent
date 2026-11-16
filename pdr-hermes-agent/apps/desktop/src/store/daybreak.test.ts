import { beforeEach, describe, expect, it, vi } from 'vitest'

import { $daybreakByProfile, daybreakKey, setDaybreak, syncDaybreak } from './daybreak'

beforeEach(() => {
  $daybreakByProfile.set({})
})

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void

  const promise = new Promise<T>((done, fail) => {
    resolve = done
    reject = fail
  })

  return { promise, reject, resolve }
}

describe('the Daybreak profile switch', () => {
  it('reads and writes agent.daybreak through the profile-scoped config keys', async () => {
    const request = vi.fn(async (method: string, params?: Record<string, unknown>) =>
      method === 'config.get' ? { value: true } : { key: 'daybreak', value: params?.value }
    )

    await syncDaybreak(request, 'work')
    expect(request).toHaveBeenCalledWith('config.get', { key: 'daybreak', profile: 'work' })
    expect($daybreakByProfile.get()).toEqual({ [daybreakKey('work')]: true })

    expect(await setDaybreak(request, 'work', false)).toBe(true)
    expect(request).toHaveBeenLastCalledWith('config.set', { key: 'daybreak', profile: 'work', value: false })
    expect($daybreakByProfile.get()).toEqual({ [daybreakKey('work')]: false })
  })

  it('restores the confirmed value when the owner refuses the write', async () => {
    await syncDaybreak(async () => ({ value: false }), '')

    const request = vi.fn(async () => {
      throw new Error('permission_denied')
    })

    const pending = setDaybreak(request, '', true)
    expect($daybreakByProfile.get()[daybreakKey('')]).toBe(true) // optimistic
    expect(await pending).toBe(false)
    expect($daybreakByProfile.get()[daybreakKey('')]).toBe(false)
    expect(request).toHaveBeenCalledWith('config.set', { key: 'daybreak', value: true }) // blank = launch profile
  })

  it('treats anything but an explicit true as off', async () => {
    await syncDaybreak(async () => ({ value: 'yes' }), 'loose')
    expect($daybreakByProfile.get()[daybreakKey('loose')]).toBe(false)
  })

  it('keeps the same profile name on two connections apart', async () => {
    await syncDaybreak(async () => ({ value: true }), 'default', 'local')
    await syncDaybreak(async () => ({ value: false }), 'default', 'remote')

    expect($daybreakByProfile.get()[daybreakKey('default', 'local')]).toBe(true)
    expect($daybreakByProfile.get()[daybreakKey('default', 'remote')]).toBe(false)
  })

  it('never lets a read that raced a write discard the write', async () => {
    const write = deferred<{ value: boolean }>()
    const read = deferred<{ value: boolean }>()
    const request = vi.fn((method: string) => (method === 'config.set' ? write.promise : read.promise))

    const setting = setDaybreak(request, 'race', true)
    const syncing = syncDaybreak(request, 'race')

    read.resolve({ value: false }) // the pre-write value
    await syncing
    expect($daybreakByProfile.get()[daybreakKey('race')]).toBe(true)

    write.resolve({ value: true })
    expect(await setting).toBe(true)
    expect($daybreakByProfile.get()[daybreakKey('race')]).toBe(true)
  })
  it('sends quick flips in order and keeps a saved flip when a later one is refused', async () => {
    await syncDaybreak(async () => ({ value: false }), 'quick')

    const turnOn = deferred<{ value: boolean }>()
    const turnOff = deferred<{ value: boolean }>()
    const replies = [turnOn, turnOff]
    const request = vi.fn(() => replies.shift()!.promise)

    const turningOn = setDaybreak(request, 'quick', true)
    const turningOff = setDaybreak(request, 'quick', false)
    expect($daybreakByProfile.get()[daybreakKey('quick')]).toBe(false) // optimistic: the latest flip
    await Promise.resolve()
    expect(request).toHaveBeenCalledTimes(1) // the off write waits for the on write's reply

    turnOn.resolve({ value: true })
    expect(await turningOn).toBe(true)
    await vi.waitFor(() => expect(request).toHaveBeenCalledTimes(2))
    expect(request).toHaveBeenLastCalledWith('config.set', { key: 'daybreak', profile: 'quick', value: false })
    expect($daybreakByProfile.get()[daybreakKey('quick')]).toBe(false) // the off flip is still pending

    turnOff.reject(new Error('refused'))
    expect(await turningOff).toBe(false)
    expect($daybreakByProfile.get()[daybreakKey('quick')]).toBe(true) // the owner saved on
  })
})
