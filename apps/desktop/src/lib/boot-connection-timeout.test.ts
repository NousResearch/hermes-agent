import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import type { DesktopBootstrapEvent, DesktopBootstrapState } from '@/global'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS } from '@/lib/with-timeout'

import { INITIAL_BOOT_MAX_WAIT_MS, waitForBootConnection } from './boot-connection-timeout'

describe('initial boot connection deadline', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  function setup(active = true, setupChoice = false) {
    let listener: ((event: DesktopBootstrapEvent) => void) | undefined

    const off = vi.fn(() => {
      listener = undefined
    })

    const bridge = {
      getBootstrapState: async () =>
        ({ active, setupChoice: setupChoice ? {} : null, stages: {} }) as DesktopBootstrapState,
      onBootstrapEvent: (callback: (event: DesktopBootstrapEvent) => void) => {
        listener = callback

        return off
      }
    }

    const controller = new AbortController()
    const result = waitForBootConnection(new Promise<never>(() => undefined), bridge, controller.signal)
    const error = result.catch(err => err as Error)

    return { controller, error, off, emit: (event: DesktopBootstrapEvent) => listener?.(event) }
  }

  it('fails when installer progress stops, despite empty logs and duplicate stage events', async () => {
    const boot = setup()
    await vi.advanceTimersByTimeAsync(1)
    boot.emit({ type: 'stage', name: 'install', state: 'running' })
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS - 1)
    boot.emit({ type: 'log', line: '  ' })
    boot.emit({ type: 'stage', name: 'install', state: 'running' })
    await vi.advanceTimersByTimeAsync(1)
    expect((await boot.error).message).toBe('Timed out connecting to Hermes backend')
    expect(boot.off).toHaveBeenCalledOnce()
  })

  it('bounds even continuous installer output with an absolute deadline', async () => {
    const boot = setup()
    await vi.advanceTimersByTimeAsync(1)

    for (let elapsed = 1; elapsed < INITIAL_BOOT_MAX_WAIT_MS; elapsed += 60_000) {
      boot.emit({ type: 'log', line: 'working' })
      await vi.advanceTimersByTimeAsync(Math.min(60_000, INITIAL_BOOT_MAX_WAIT_MS - elapsed))
    }

    expect((await boot.error).message).toBe('Timed out connecting to Hermes backend')
    expect(boot.off).toHaveBeenCalledOnce()
  })

  it('bounds time spent at setup choice by the absolute deadline', async () => {
    const boot = setup(false, true)
    await vi.advanceTimersByTimeAsync(INITIAL_BOOT_MAX_WAIT_MS)
    expect((await boot.error).message).toBe('Timed out connecting to Hermes backend')
  })

  it('resumes the idle deadline when the user leaves setup choice', async () => {
    const boot = setup(false, true)
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS + 1)
    boot.emit({ type: 'setup-choice', active: false })
    await vi.advanceTimersByTimeAsync(BACKEND_BOOT_WAIT_TIMEOUT_MS)
    expect((await boot.error).message).toBe('Timed out connecting to Hermes backend')
  })

  it('detaches the listener and clears timers on unmount', async () => {
    const boot = setup()
    boot.controller.abort()
    expect((await boot.error).message).toBe('Boot connection wait cancelled')
    expect(boot.off).toHaveBeenCalledOnce()
    expect(vi.getTimerCount()).toBe(0)
  })
})
