/**
 * WSL availability gate (#128770): on a Windows host with no registered WSL
 * distribution, the bridge must never launch `wsl.exe` or touch a `\\wsl$` /
 * `\\wsl.localhost` share. Both wake the WSL installer stub and pop an
 * "install WSL" console. Availability comes from the registry only.
 */
import assert from 'node:assert/strict'

import { afterEach, beforeEach, test, vi } from 'vitest'

const HKCU = 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Lxss'
const HKLM = 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Lxss'
const registration = (root: string) => `${root}\\{distribution-guid}\n    DistributionName    REG_SZ    Ubuntu\n`

interface Call {
  args: string[]
  exe: string
  options: Record<string, unknown>
}

const io = vi.hoisted(() => ({
  calls: [] as Call[],
  probes: [] as string[],
  registry: {} as Record<string, Error | string>
}))

vi.mock('node:child_process', () => ({
  execFileSync: (exe: string, args: string[], options: Record<string, unknown>) => {
    io.calls.push({ args, exe, options })

    if (exe === 'wsl.exe') {
      return 'Ubuntu\n'
    }

    const result = io.registry[args[1]]

    if (result === undefined || result instanceof Error) {
      throw result ?? new Error('missing key')
    }

    return result
  }
}))

vi.mock('node:fs', () => {
  const existsSync = (value: string) => {
    io.probes.push(value)

    return true
  }

  return { default: { existsSync }, existsSync }
})

const originalPlatform = process.platform

async function loadBridge(registry: Record<string, Error | string>, platform: NodeJS.Platform = 'win32') {
  io.registry = registry
  Object.defineProperty(process, 'platform', { value: platform })
  vi.resetModules()

  return import('./wsl-path-bridge')
}

beforeEach(() => {
  io.calls.length = 0
  io.probes.length = 0
})

afterEach(() => {
  Object.defineProperty(process, 'platform', { value: originalPlatform })
})

const cases: [string, Record<string, Error | string>, boolean][] = [
  ['per-user distribution with no machine key', { [HKCU]: registration(HKCU) }, true],
  ['machine distribution fallback', { [HKLM]: registration(HKLM) }, true],
  ['full user hive spelling from reg.exe', { [HKCU]: registration(HKCU.replace('HKCU', 'HKEY_CURRENT_USER')) }, true],
  ['full machine hive spelling from reg.exe', { [HKLM]: registration(HKLM.replace('HKLM', 'HKEY_LOCAL_MACHINE')) }, true],
  ['missing keys', {}, false],
  ['empty machine shell', { [HKLM]: `${HKLM}\n` }, false],
  ['empty user shell and stale default', { [HKCU]: `${HKCU}\n    DefaultDistribution    REG_SZ    {stale}\n` }, false],
  ['empty distribution name', { [HKCU]: `${HKCU}\\{guid}\n    DistributionName    REG_SZ    \n` }, false],
  ['value on root instead of a distribution subkey', { [HKCU]: `${HKCU}\n    DistributionName    REG_SZ    Ubuntu\n` }, false],
  ['registry failure', { [HKCU]: new Error('timeout'), [HKLM]: new Error('access denied') }, false]
]

for (const [name, registry, available] of cases) {
  test(`Windows availability gate: ${name}`, async () => {
    const bridge = await loadBridge(registry)

    assert.equal(bridge.isWslBridgeActive('unknown'), available)
    bridge.setWslBridgeProfileState('local', true)
    assert.equal(bridge.isWslBridgeActive('local'), available)

    if (available) {
      assert.equal(bridge.resolveLocalReadPath('/home/alex', undefined, 'local'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
      assert.equal(
        bridge.resolvePickerDefaultPath('/home/alex', undefined, 'local'),
        '\\\\wsl.localhost\\Ubuntu\\home\\alex'
      )
    } else {
      // POSIX paths stay untouched: no distro lookup, no UNC share probe.
      assert.equal(bridge.resolveLocalReadPath('/home/alex', undefined, 'local'), '/home/alex')
      assert.equal(bridge.resolvePickerDefaultPath('/home/alex', undefined, 'local'), '/home/alex')
      // Drive aliases still map to the local native path.
      assert.equal(bridge.resolveLocalReadPath('/mnt/c/Users/alex', undefined, 'local'), 'C:/Users/alex')
      assert.ok(io.probes.every(probe => !probe.startsWith('\\\\')))
      assert.ok(io.calls.every(call => call.exe === 'reg.exe'))
    }

    const registryCalls = io.calls.filter(call => call.exe === 'reg.exe')
    assert.equal(registryCalls[0].args[1], HKCU)
    assert.ok(registryCalls.length <= 2, 'availability is probed once and cached')

    for (const { options } of registryCalls) {
      assert.equal(options.windowsHide, true)
      assert.equal(options.timeout, 1000)
      assert.deepEqual(options.stdio, ['ignore', 'pipe', 'ignore'])
    }
  })
}

test('remote profiles short-circuit without a registry query', async () => {
  const bridge = await loadBridge({ [HKCU]: registration(HKCU) })

  bridge.setWslBridgeProfileState('remote', false)
  assert.equal(bridge.resolvePickerDefaultPath('/home/alex', undefined, 'remote'), '/home/alex')
  assert.deepEqual(io.calls, [])
})

test('non-Windows keeps profile behaviour without registry calls', async () => {
  const bridge = await loadBridge({}, 'darwin')

  assert.equal(bridge.isWslBridgeActive(), true)
  bridge.setWslBridgeProfileState('remote', false)
  assert.equal(bridge.isWslBridgeActive('remote'), false)
  assert.deepEqual(io.calls, [])
})
