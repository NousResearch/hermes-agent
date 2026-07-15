import { describe, expect, it, vi } from 'vitest'

import {
  createPowerSaveBlockerController,
  isPowerSaveRefreshAuthorized,
  powerSaveTypeForMode,
  resolvePreventSleepConfig
} from './power-save'

describe('resolvePreventSleepConfig', () => {
  it('is disabled with system mode by default', () => {
    expect(resolvePreventSleepConfig(undefined)).toEqual({
      enabled: false,
      mode: 'system',
      surfaces: []
    })
  })

  it('accepts the canonical resolved payload', () => {
    expect(
      resolvePreventSleepConfig({
        enabled: true,
        mode: 'display',
        surfaces: ['desktop', 'gateway']
      })
    ).toEqual({
      enabled: true,
      mode: 'display',
      surfaces: ['desktop', 'gateway']
    })
  })

  it('fails closed instead of duplicating backend normalization', () => {
    expect(resolvePreventSleepConfig({ enabled: 'true', surfaces: ['desktop'] }).enabled).toBe(false)
    expect(resolvePreventSleepConfig(true, 'gateway').enabled).toBe(false)
    expect(resolvePreventSleepConfig({ enabled: true, mode: 'invalid' }).mode).toBe('system')
  })

  it('treats an explicit empty surface list as disabled everywhere', () => {
    expect(resolvePreventSleepConfig({ enabled: true, surfaces: [] }, 'desktop').enabled).toBe(false)
    expect(resolvePreventSleepConfig({ enabled: true, surfaces: [] }, 'gateway').enabled).toBe(false)
  })
})

describe('powerSaveTypeForMode', () => {
  it('keeps the display awake only when explicitly requested', () => {
    expect(powerSaveTypeForMode('system')).toBe('prevent-app-suspension')
    expect(powerSaveTypeForMode('display')).toBe('prevent-display-sleep')
  })
})

describe('isPowerSaveRefreshAuthorized', () => {
  it('allows only the primary renderer that owns the machine-wide effect', () => {
    const mainRenderer = {}

    expect(isPowerSaveRefreshAuthorized(mainRenderer, mainRenderer)).toBe(true)
    expect(isPowerSaveRefreshAuthorized({}, mainRenderer)).toBe(false)
    expect(isPowerSaveRefreshAuthorized(mainRenderer, null)).toBe(false)
  })
})

describe('createPowerSaveBlockerController', () => {
  it('starts, avoids duplicate starts, switches mode, and stops', () => {
    let nextId = 40
    const active = new Set<number>()

    const powerSaveBlocker = {
      start: vi.fn((type: string) => {
        void type
        const id = ++nextId
        active.add(id)

        return id
      }),
      stop: vi.fn((id: number) => active.delete(id)),
      isStarted: vi.fn((id: number) => active.has(id))
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    expect(controller.refresh({ enabled: true, surfaces: ['desktop'], mode: 'system' })).toMatchObject({
      active: true,
      changed: true,
      mode: 'system'
    })
    expect(powerSaveBlocker.start).toHaveBeenCalledWith('prevent-app-suspension')

    expect(controller.refresh({ enabled: true, surfaces: ['desktop'], mode: 'system' }).changed).toBe(false)
    expect(powerSaveBlocker.start).toHaveBeenCalledTimes(1)

    expect(controller.refresh({ enabled: true, surfaces: ['desktop'], mode: 'display' })).toMatchObject({
      active: true,
      changed: true,
      mode: 'display'
    })
    expect(powerSaveBlocker.stop).toHaveBeenCalledWith(41)
    expect(powerSaveBlocker.start).toHaveBeenLastCalledWith('prevent-display-sleep')
    expect(powerSaveBlocker.start.mock.invocationCallOrder[1]).toBeLessThan(
      powerSaveBlocker.stop.mock.invocationCallOrder[0]
    )

    expect(controller.stop()).toBe(true)
    expect(controller.stop()).toBe(false)
    expect(controller.state().active).toBe(false)
  })

  it('is a no-op when disabled for the desktop surface', () => {
    const powerSaveBlocker = {
      start: vi.fn(() => 1),
      stop: vi.fn(() => true),
      isStarted: vi.fn(() => false)
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    expect(controller.refresh({ enabled: true, surfaces: ['gateway'] })).toMatchObject({
      active: false,
      changed: false
    })
    expect(powerSaveBlocker.start).not.toHaveBeenCalled()
  })

  it('recovers if Electron reports that the previous blocker is no longer active', () => {
    let active = true

    const powerSaveBlocker = {
      start: vi.fn(() => {
        active = true

        return 7
      }),
      stop: vi.fn(() => true),
      isStarted: vi.fn(() => active)
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    controller.refresh({ enabled: true, surfaces: ['desktop'] })
    active = false
    controller.refresh({ enabled: true, surfaces: ['desktop'] })

    expect(powerSaveBlocker.start).toHaveBeenCalledTimes(2)
  })

  it('keeps the last-good blocker when replacement acquisition fails', () => {
    const active = new Set<number>()

    const powerSaveBlocker = {
      start: vi
        .fn()
        .mockImplementationOnce(() => {
          active.add(1)

          return 1
        })
        .mockImplementationOnce(() => {
          throw new Error('native start failed')
        }),
      stop: vi.fn((id: number) => active.delete(id)),
      isStarted: vi.fn((id: number) => active.has(id))
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    controller.refresh({ enabled: true, mode: 'system', surfaces: ['desktop'] })

    expect(() => controller.refresh({ enabled: true, mode: 'display', surfaces: ['desktop'] })).toThrow(
      'native start failed'
    )
    expect(controller.state()).toMatchObject({ active: true, mode: 'system' })
    expect(active).toEqual(new Set([1]))
    expect(powerSaveBlocker.stop).not.toHaveBeenCalled()
  })

  it('keeps the last-good blocker when Electron returns an inactive replacement', () => {
    const active = new Set<number>()

    const powerSaveBlocker = {
      start: vi
        .fn()
        .mockImplementationOnce(() => {
          active.add(1)

          return 1
        })
        .mockReturnValueOnce(2),
      stop: vi.fn((id: number) => active.delete(id)),
      isStarted: vi.fn((id: number) => active.has(id))
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    controller.refresh({ enabled: true, mode: 'system', surfaces: ['desktop'] })

    expect(() => controller.refresh({ enabled: true, mode: 'display', surfaces: ['desktop'] })).toThrow('did not start')
    expect(controller.state()).toMatchObject({ active: true, mode: 'system' })
    expect(active).toEqual(new Set([1]))
  })

  it('rolls back a replacement when the previous blocker cannot be released', () => {
    let nextId = 0
    const active = new Set<number>()

    const powerSaveBlocker = {
      start: vi.fn(() => {
        const id = ++nextId

        active.add(id)

        return id
      }),
      stop: vi.fn((id: number) => (id === 1 ? false : active.delete(id))),
      isStarted: vi.fn((id: number) => active.has(id))
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    controller.refresh({ enabled: true, mode: 'system', surfaces: ['desktop'] })

    expect(() => controller.refresh({ enabled: true, mode: 'display', surfaces: ['desktop'] })).toThrow(
      'Failed to replace'
    )
    expect(controller.state()).toMatchObject({ active: true, mode: 'system' })
    expect(active).toEqual(new Set([1]))
  })

  it('retains a failed blocker release so a later refresh can retry it', () => {
    let canStop = false
    const active = new Set([1])

    const powerSaveBlocker = {
      start: vi.fn(() => 1),
      stop: vi.fn((id: number) => canStop && active.delete(id)),
      isStarted: vi.fn((id: number) => active.has(id))
    }

    const controller = createPowerSaveBlockerController(powerSaveBlocker)

    controller.refresh({ enabled: true, surfaces: ['desktop'] })

    expect(() => controller.refresh({ enabled: false })).toThrow('Failed to stop')
    expect(controller.state().active).toBe(true)

    canStop = true
    expect(controller.refresh({ enabled: false })).toMatchObject({ active: false, changed: true })
    expect(powerSaveBlocker.stop).toHaveBeenCalledTimes(2)
  })
})
