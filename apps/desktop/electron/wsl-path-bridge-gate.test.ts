/**
 * Windows-platform regression for the WSL path-bridge gate (#66433).
 *
 * The behavioural tests in wsl-path-bridge.test.ts prove the no-op contract
 * (paths pass through unchanged when the bridge is inactive). This file goes
 * one rung further: with `process.platform` stubbed to `win32` and
 * `child_process.execFileSync` mocked, it proves the actual `wsl.exe` spawn is
 * suppressed — not just that the return value looks right.
 *
 * Each test re-imports the module fresh (vi.resetModules) so IS_WINDOWS is
 * re-evaluated against the stubbed platform.
 */
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

const execFileSyncMock = vi.fn(() => 'Ubuntu\n')

vi.mock('node:child_process', () => ({ execFileSync: execFileSyncMock }))

describe('WSL bridge gate on Windows (#66433)', () => {
  const realPlatform = process.platform

  beforeEach(() => {
    Object.defineProperty(process, 'platform', { value: 'win32', configurable: true })
    vi.resetModules()
    execFileSyncMock.mockClear()
  })

  afterEach(() => {
    Object.defineProperty(process, 'platform', { value: realPlatform, configurable: true })
  })

  test('wsl.exe IS probed for a POSIX path when the bridge is active (control)', async () => {
    const { resolveLocalReadPath } = await import('./wsl-path-bridge')
    resolveLocalReadPath('/home/ubuntu/project')
    expect(execFileSyncMock).toHaveBeenCalled()
    // Sanity: it really was wsl.exe, not some other binary.
    expect(execFileSyncMock).toHaveBeenNthCalledWith(
      1,
      'wsl.exe',
      expect.arrayContaining(['-l', '-q']),
      expect.anything()
    )
  })

  test('wsl.exe is NEVER probed when the bridge is inactive — even for POSIX paths', async () => {
    const { resolveLocalReadPath, setWslBridgeActive } = await import('./wsl-path-bridge')
    setWslBridgeActive(false)
    // A POSIX path that WOULD trigger bridging (and the wsl.exe probe) when
    // active — but with the bridge off, resolveDefaultWslDistro is never
    // reached because resolveLocalReadPath returns before it.
    const result = resolveLocalReadPath('/home/ubuntu/project')
    expect(execFileSyncMock).not.toHaveBeenCalled()
    expect(result).toBe('/home/ubuntu/project')
  })

  test('the picker default-path also skips the wsl.exe probe when inactive', async () => {
    const { resolvePickerDefaultPath, setWslBridgeActive } = await import('./wsl-path-bridge')
    setWslBridgeActive(false)
    const result = resolvePickerDefaultPath('/home/ubuntu')
    expect(execFileSyncMock).not.toHaveBeenCalled()
    expect(result).toBe('/home/ubuntu')
  })

  test('re-enabling the bridge restores wsl.exe probing', async () => {
    const { resolveLocalReadPath, setWslBridgeActive } = await import('./wsl-path-bridge')
    setWslBridgeActive(false)
    resolveLocalReadPath('/home/ubuntu/project')
    expect(execFileSyncMock).not.toHaveBeenCalled()

    setWslBridgeActive(true)
    execFileSyncMock.mockClear()
    resolveLocalReadPath('/home/ubuntu/project')
    expect(execFileSyncMock).toHaveBeenCalled()
  })
})
