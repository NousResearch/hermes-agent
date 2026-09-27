import { describe, expect, it } from 'vitest'

import { decideNvidiaEglFallback, NVIDIA_BROKEN_EGL_MAJOR, parseNvidiaDriverMajor } from './linux-nvidia-egl-fallback'

const LINUX = { env: {}, platform: 'linux' as const, isWsl: false, remoteDisplayReason: null }

describe('parseNvidiaDriverMajor', () => {
  it('parses the major from /proc/driver/nvidia/version content', () => {
    const text = 'NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.82.09  Mon Jul 21 19:44:16 UTC 2025\n'

    expect(parseNvidiaDriverMajor(text)).toBe(580)
  })

  it('parses 3-digit majors like 570 and 550', () => {
    expect(parseNvidiaDriverMajor('NVRM version: ...  570.133.07 ...\n')).toBe(570)
    expect(parseNvidiaDriverMajor('NVRM version: ...  550.107.02 ...\n')).toBe(550)
  })

  it('returns null for garbage or empty input', () => {
    expect(parseNvidiaDriverMajor('')).toBeNull()
    expect(parseNvidiaDriverMajor('no driver version here')).toBeNull()
  })
})

describe('decideNvidiaEglFallback', () => {
  it('enables on linux with driver >= 580', () => {
    const decision = decideNvidiaEglFallback({ ...LINUX, driverMajor: 580 })
    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('580')
  })

  it('enables on newer majors (570-series is fine, 580+ is not)', () => {
    expect(decideNvidiaEglFallback({ ...LINUX, driverMajor: 570 }).enable).toBe(false)
    expect(decideNvidiaEglFallback({ ...LINUX, driverMajor: 580 + 5 }).enable).toBe(true)
    expect(NVIDIA_BROKEN_EGL_MAJOR).toBe(580)
  })

  it('stays off below the broken major and when detection finds no driver', () => {
    expect(decideNvidiaEglFallback({ ...LINUX, driverMajor: 570 }).enable).toBe(false)
    expect(decideNvidiaEglFallback({ ...LINUX, driverMajor: null }).enable).toBe(false)
  })

  it('stays off on non-linux platforms', () => {
    expect(decideNvidiaEglFallback({ ...LINUX, platform: 'darwin', driverMajor: 580 }).enable).toBe(false)
    expect(decideNvidiaEglFallback({ ...LINUX, platform: 'win32', driverMajor: 580 }).enable).toBe(false)
  })

  it('stays off under WSLg', () => {
    expect(decideNvidiaEglFallback({ ...LINUX, isWsl: true, driverMajor: 580 }).enable).toBe(false)
  })

  it('stays off when a remote display already forced software rendering', () => {
    expect(
      decideNvidiaEglFallback({
        ...LINUX,
        driverMajor: 580,
        remoteDisplayReason: 'ssh-session'
      }).enable
    ).toBe(false)
  })

  it('HERMES_DESKTOP_DISABLE_GPU=0 keeps the GPU (and the fallback) off', () => {
    expect(
      decideNvidiaEglFallback({
        ...LINUX,
        driverMajor: 580,
        env: { HERMES_DESKTOP_DISABLE_GPU: '0' }
      }).enable
    ).toBe(false)
  })

  it('HERMES_DESKTOP_NVIDIA_SWIFTSHADER forces the fallback on without detection', () => {
    const decision = decideNvidiaEglFallback({
      ...LINUX,
      driverMajor: null,
      env: { HERMES_DESKTOP_NVIDIA_SWIFTSHADER: '1' }
    })

    expect(decision.enable).toBe(true)
    expect(decision.reason).toContain('override')
  })

  it('HERMES_DESKTOP_NVIDIA_SWIFTSHADER=0 opts out even on affected drivers', () => {
    expect(
      decideNvidiaEglFallback({
        ...LINUX,
        driverMajor: 580,
        env: { HERMES_DESKTOP_NVIDIA_SWIFTSHADER: 'off' }
      }).enable
    ).toBe(false)
  })
})
