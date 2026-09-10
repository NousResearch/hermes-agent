import { describe, expect, it } from 'vitest'

import {
  $desktopBoot,
  applyDesktopBootProgress,
  completeDesktopBoot,
  failDesktopBoot,
  resumeDesktopBootForRetry
} from './boot'

// #92927: the dead-IPC-bridge boot failure must be distinguishable from a
// generic boot failure so the recovery overlay can stop offering
// bridge-dependent dead ends (Repair / Settings / Open logs all round-trip
// through window.hermesDesktop) and instead show the actionable repair copy.

describe('boot store errorCode (#92927)', () => {
  it('applyDesktopBootProgress carries the stable main-process errorCode', () => {
    applyDesktopBootProgress({
      error: 'Local Hermes backend is reachable but its JSON-RPC gateway did not answer.',
      errorCode: 'gateway.rpc-probe-failed',
      fakeMode: false,
      message: 'Desktop boot failed',
      phase: 'backend.error',
      progress: 90,
      running: false,
      timestamp: Date.now()
    })

    expect($desktopBoot.get().errorCode).toBe('gateway.rpc-probe-failed')
  })

  it('a running progress update cannot clobber a latched failure errorCode', () => {
    applyDesktopBootProgress({
      error: 'Local Hermes backend is reachable but its JSON-RPC gateway did not answer.',
      errorCode: 'gateway.rpc-probe-failed',
      fakeMode: false,
      message: 'Desktop boot failed',
      phase: 'backend.error',
      progress: 90,
      running: false,
      timestamp: Date.now()
    })

    applyDesktopBootProgress({
      error: null,
      fakeMode: false,
      message: 'backend.ready',
      phase: 'backend.ready',
      progress: 94,
      running: true,
      timestamp: Date.now()
    })

    expect($desktopBoot.get().errorCode).toBe('gateway.rpc-probe-failed')
  })

  it('failDesktopBoot drops a stale main-process errorCode (renderer failures carry none)', () => {
    applyDesktopBootProgress({
      error: 'Local Hermes backend is reachable but its JSON-RPC gateway did not answer.',
      errorCode: 'gateway.rpc-probe-failed',
      fakeMode: false,
      message: 'Desktop boot failed',
      phase: 'backend.error',
      progress: 90,
      running: false,
      timestamp: Date.now()
    })

    failDesktopBoot('Desktop IPC bridge is unavailable.', 'ipc-bridge')

    expect($desktopBoot.get().errorCode).toBeUndefined()
    expect($desktopBoot.get().errorKind).toBe('ipc-bridge')
  })

  it('completeDesktopBoot clears the failure code', () => {
    applyDesktopBootProgress({
      error: 'Local Hermes backend is reachable but its JSON-RPC gateway did not answer.',
      errorCode: 'gateway.rpc-probe-failed',
      fakeMode: false,
      message: 'Desktop boot failed',
      phase: 'backend.error',
      progress: 90,
      running: false,
      timestamp: Date.now()
    })
    completeDesktopBoot()

    expect($desktopBoot.get().error).toBeNull()
    expect($desktopBoot.get().errorCode).toBeUndefined()
  })
})

describe('boot store errorKind (#92927)', () => {
  it('failDesktopBoot records the failure kind', () => {
    failDesktopBoot('Desktop IPC bridge is unavailable.', 'ipc-bridge')

    expect($desktopBoot.get().error).toBe('Desktop IPC bridge is unavailable.')
    expect($desktopBoot.get().errorKind).toBe('ipc-bridge')
  })

  it('failDesktopBoot without a kind leaves errorKind undefined', () => {
    failDesktopBoot('Hermes backend exited before it became ready.')

    expect($desktopBoot.get().errorKind).toBeUndefined()
  })

  it('a later boot-progress event cannot clobber the latched failure kind', () => {
    failDesktopBoot('Desktop IPC bridge is unavailable.', 'ipc-bridge')

    applyDesktopBootProgress({
      error: null,
      fakeMode: false,
      message: 'backend.ready',
      phase: 'backend.ready',
      progress: 94,
      running: true,
      timestamp: Date.now()
    })

    expect($desktopBoot.get().error).toBe('Desktop IPC bridge is unavailable.')
    expect($desktopBoot.get().errorKind).toBe('ipc-bridge')
  })

  it('completeDesktopBoot clears the failure kind', () => {
    failDesktopBoot('Desktop IPC bridge is unavailable.', 'ipc-bridge')
    completeDesktopBoot()

    expect($desktopBoot.get().error).toBeNull()
    expect($desktopBoot.get().errorKind).toBeUndefined()
  })

  it('resumeDesktopBootForRetry clears the failure kind while retrying', () => {
    failDesktopBoot('Desktop IPC bridge is unavailable.', 'ipc-bridge')
    resumeDesktopBootForRetry('retrying…')

    expect($desktopBoot.get().error).toBeNull()
    expect($desktopBoot.get().errorKind).toBeUndefined()
  })
})
