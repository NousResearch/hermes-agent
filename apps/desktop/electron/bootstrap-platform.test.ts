import assert from 'node:assert/strict'
import fs from 'node:fs'

import { test, vi } from 'vitest'

import {
  bundledRuntimeImportCheck,
  detectRemoteDisplay,
  isVirtualizedGpuEnvironment,
  isWindowsBinaryPathInWsl,
  isWslEnvironment
} from './bootstrap-platform'

test('isWslEnvironment detects WSL2 env vars on linux', () => {
  assert.equal(isWslEnvironment({ WSL_DISTRO_NAME: 'Ubuntu' }, 'linux'), true)
  assert.equal(isWslEnvironment({ WSL_INTEROP: '/run/WSL/123_interop' }, 'linux'), true)
  assert.equal(isWslEnvironment({}, 'linux', '6.6.87.2-microsoft-standard-WSL2'), true)
  assert.equal(isWslEnvironment({}, 'linux', '6.6.87-generic'), false)
  assert.equal(isWslEnvironment({ WSL_DISTRO_NAME: 'Ubuntu' }, 'darwin'), false)
})

test('isVirtualizedGpuEnvironment detects QEMU via injected DMI vendor', () => {
  assert.match(String(isVirtualizedGpuEnvironment('linux', 'QEMU', '')), /QEMU detected/)
  assert.equal(isVirtualizedGpuEnvironment('linux', 'LENOVO', ''), null)
  assert.equal(isVirtualizedGpuEnvironment('darwin', 'QEMU', ''), null)
})

test('isVirtualizedGpuEnvironment detects Apple Virtualization.framework via injected product name', () => {
  assert.match(
    String(isVirtualizedGpuEnvironment('linux', 'Apple Inc.', 'Apple Virtualization Generic Platform')),
    /Apple Virtualization\.framework detected/
  )
  assert.equal(isVirtualizedGpuEnvironment('linux', 'Apple Inc.', 'MacBookPro18,3'), null)
})

test('isVirtualizedGpuEnvironment returns null when the sysfs read fails', () => {
  const spy = vi.spyOn(fs, 'readFileSync').mockImplementation(() => {
    throw new Error('ENOENT')
  })

  assert.equal(isVirtualizedGpuEnvironment('linux'), null)

  spy.mockRestore()
})

test('isWindowsBinaryPathInWsl blocks Windows binary types on WSL', () => {
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.exe', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.cmd', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.bat', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/install.ps1', { isWsl: true }), true)
  assert.equal(isWindowsBinaryPathInWsl('/usr/local/bin/hermes', { isWsl: true }), false)
  assert.equal(isWindowsBinaryPathInWsl('/mnt/c/Tools/hermes.exe', { isWsl: false }), false)
})

test('bundledRuntimeImportCheck selects platform-specific import checks', () => {
  assert.equal(bundledRuntimeImportCheck('win32'), 'import fastapi, uvicorn, winpty')
  assert.equal(bundledRuntimeImportCheck('darwin'), 'import fastapi, uvicorn, ptyprocess')
  assert.equal(bundledRuntimeImportCheck('linux'), 'import fastapi, uvicorn, ptyprocess')
})

test('detectRemoteDisplay keeps GPU on for local sessions', () => {
  // Plain local X11, Wayland, native Windows, native macOS — no remote signal.
  // sysVendor/productName are explicitly injected as empty so this test is
  // isolated from whatever real hardware/VM it happens to run on.
  assert.equal(
    detectRemoteDisplay({ env: { DISPLAY: ':0' }, platform: 'linux', sysVendor: '', productName: '' }),
    null
  )
  assert.equal(
    detectRemoteDisplay({ env: { WAYLAND_DISPLAY: 'wayland-0' }, platform: 'linux', sysVendor: '', productName: '' }),
    null
  )
  assert.equal(detectRemoteDisplay({ env: { SESSIONNAME: 'Console' }, platform: 'win32' }), null)
  assert.equal(detectRemoteDisplay({ env: {}, platform: 'darwin' }), null)
})

test('detectRemoteDisplay does not treat WSLg as remote', () => {
  // WSLg renders locally via vGPU and doesn't show the flicker, so a WSL
  // session with a local DISPLAY keeps hardware acceleration on.
  // sysVendor/productName injected as empty to isolate from real host hardware.
  assert.equal(
    detectRemoteDisplay({
      env: { WSL_DISTRO_NAME: 'Ubuntu', DISPLAY: ':0' },
      platform: 'linux',
      sysVendor: '',
      productName: ''
    }),
    null
  )
  assert.equal(
    detectRemoteDisplay({
      env: { WSL_INTEROP: '/run/WSL/1_interop', DISPLAY: ':0' },
      platform: 'linux',
      sysVendor: '',
      productName: ''
    }),
    null
  )
})

test('detectRemoteDisplay flags SSH sessions on any platform', () => {
  assert.equal(
    detectRemoteDisplay({ env: { SSH_CONNECTION: '1.2.3.4 5 6.7.8.9 22' }, platform: 'linux' }),
    'ssh-session'
  )
  assert.equal(detectRemoteDisplay({ env: { SSH_CLIENT: '1.2.3.4 5 22' }, platform: 'darwin' }), 'ssh-session')
  assert.equal(detectRemoteDisplay({ env: { SSH_TTY: '/dev/pts/0' }, platform: 'win32' }), 'ssh-session')
})

test('detectRemoteDisplay flags forwarded X11 displays but not local ones', () => {
  assert.match(String(detectRemoteDisplay({ env: { DISPLAY: 'localhost:10.0' }, platform: 'linux' })), /x11-forwarding/)
  assert.match(String(detectRemoteDisplay({ env: { DISPLAY: '192.168.1.5:0' }, platform: 'linux' })), /x11-forwarding/)
  assert.equal(
    detectRemoteDisplay({ env: { DISPLAY: ':1' }, platform: 'linux', sysVendor: '', productName: '' }),
    null
  )
})

test('detectRemoteDisplay flags RDP sessions', () => {
  assert.match(String(detectRemoteDisplay({ env: { SESSIONNAME: 'RDP-Tcp#7' }, platform: 'win32' })), /^rdp/)
})

test('detectRemoteDisplay honors the HERMES_DESKTOP_DISABLE_GPU override both ways', () => {
  // Force-on even on a local display.
  assert.match(
    String(detectRemoteDisplay({ env: { HERMES_DESKTOP_DISABLE_GPU: '1', DISPLAY: ':0' }, platform: 'linux' })),
    /override/
  )
  // Force-off even over SSH (escape hatch when a remote display has working accel).
  assert.equal(
    detectRemoteDisplay({
      env: { HERMES_DESKTOP_DISABLE_GPU: 'false', SSH_CONNECTION: '1.2.3.4 5 6.7.8.9 22' },
      platform: 'linux'
    }),
    null
  )
})

test('detectRemoteDisplay flags virtualized GPU environments via injected sysVendor', () => {
  assert.match(
    String(detectRemoteDisplay({ env: {}, platform: 'linux', sysVendor: 'QEMU', productName: '' })),
    /virtualized-gpu \(QEMU detected/
  )
  assert.equal(
    detectRemoteDisplay({ env: {}, platform: 'linux', sysVendor: 'LENOVO', productName: '' }),
    null
  )
})

test('detectRemoteDisplay flags Apple Virtualization.framework (UTM on Apple Silicon) via injected productName', () => {
  assert.match(
    String(
      detectRemoteDisplay({
        env: {},
        platform: 'linux',
        sysVendor: 'Apple Inc.',
        productName: 'Apple Virtualization Generic Platform'
      })
    ),
    /virtualized-gpu \(Apple Virtualization\.framework detected/
  )
  assert.equal(
    detectRemoteDisplay({
      env: {},
      platform: 'linux',
      sysVendor: 'Apple Inc.',
      productName: 'MacBookPro18,3'
    }),
    null
  )
})

test('detectRemoteDisplay honors HERMES_DESKTOP_DISABLE_GPU override over virtualized GPU', () => {
  assert.equal(
    detectRemoteDisplay({
      env: { HERMES_DESKTOP_DISABLE_GPU: 'false' },
      platform: 'linux',
      sysVendor: 'QEMU',
      productName: ''
    }),
    null
  )
  assert.equal(
    detectRemoteDisplay({
      env: { HERMES_DESKTOP_DISABLE_GPU: 'false' },
      platform: 'linux',
      sysVendor: 'Apple Inc.',
      productName: 'Apple Virtualization Generic Platform'
    }),
    null
  )
})
