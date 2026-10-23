// Run: node --experimental-vm-modules --test tests-js/wsl-path-bridge-availability.test.mjs
import assert from 'node:assert/strict'
import fs from 'node:fs'
import { stripTypeScriptTypes } from 'node:module'
import path from 'node:path'
import test from 'node:test'
import vm from 'node:vm'
import { fileURLToPath } from 'node:url'

const electronDir = fileURLToPath(new URL('../apps/desktop/electron/', import.meta.url))
const hkcu = 'HKCU\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Lxss'
const hklm = 'HKLM\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Lxss'
const registration = root => `${root}\\{distribution-guid}\n    DistributionName    REG_SZ    Ubuntu\n`

async function loadBridge(registry = {}, platform = 'win32') {
  const calls = []
  const probes = []
  const context = vm.createContext({ process: { platform, env: {} } })
  const child = new vm.SyntheticModule(['execFileSync'], function () {
    this.setExport('execFileSync', (exe, args, options) => {
      calls.push({ exe, args, options })
      if (exe === 'wsl.exe') return 'Ubuntu\n'
      assert.equal(exe, 'reg.exe')
      const result = registry[args[1]]
      if (result === undefined || result instanceof Error) throw result || new Error('missing key')
      return result
    })
  }, { context })
  const files = new vm.SyntheticModule(['default'], function () {
    this.setExport('default', { existsSync: value => { probes.push(value); return true } })
  }, { context })
  const source = stripTypeScriptTypes(fs.readFileSync(path.join(electronDir, 'wsl-path-bridge.ts'), 'utf8'))
  const bridge = new vm.SourceTextModule(source, { context })
  await bridge.link(specifier => specifier === 'node:child_process' ? child : files)
  await bridge.evaluate()
  return { bridge: bridge.namespace, calls, probes }
}

for (const [name, registry, expected] of [
  ['per-user distribution with no machine key', { [hkcu]: registration(hkcu) }, true],
  ['machine distribution fallback', { [hklm]: registration(hklm) }, true],
  ['full user hive spelling from reg.exe', { [hkcu]: registration(hkcu.replace('HKCU', 'HKEY_CURRENT_USER')) }, true],
  ['full machine hive spelling from reg.exe', { [hklm]: registration(hklm.replace('HKLM', 'HKEY_LOCAL_MACHINE')) }, true],
  ['missing keys', {}, false],
  ['empty machine shell', { [hklm]: `${hklm}\n` }, false],
  ['empty user shell and stale default', { [hkcu]: `${hkcu}\n    DefaultDistribution    REG_SZ    {stale}\n` }, false],
  ['empty distribution name', { [hkcu]: `${hkcu}\\{guid}\n    DistributionName    REG_SZ    \n` }, false],
  ['value on root instead of a distribution subkey', { [hkcu]: `${hkcu}\n    DistributionName    REG_SZ    Ubuntu\n` }, false],
  ['registry failure', { [hkcu]: new Error('timeout'), [hklm]: new Error('access denied') }, false]
]) {
  test(name, async () => {
    const { bridge, calls, probes } = await loadBridge(registry)
    assert.equal(bridge.isWslBridgeActive('unknown'), expected)
    bridge.setWslBridgeProfileState('local', true)
    assert.equal(bridge.isWslBridgeActive('local'), expected)
    const registryCalls = calls.length
    if (expected) {
      assert.equal(bridge.resolveLocalReadPath('/home/alex', undefined, 'local'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
      assert.equal(bridge.resolvePickerDefaultPath('/home/alex', undefined, 'local'), '\\\\wsl.localhost\\Ubuntu\\home\\alex')
      assert.equal(calls.filter(call => call.exe === 'reg.exe').length, registryCalls)
    } else {
      for (const value of ['/home/alex', '/mnt/c/Users/alex', '/C:/Users/alex']) {
        assert.equal(bridge.resolveLocalReadPath(value, undefined, 'local'), value)
        assert.equal(bridge.resolvePickerDefaultPath(value, undefined, 'local'), value)
      }
      assert.equal(calls.length, registryCalls)
      assert.deepEqual(probes, [])
    }
    bridge.setWslBridgeProfileState('remote', false)
    assert.equal(bridge.resolvePickerDefaultPath('/home/alex', undefined, 'remote'), '/home/alex')
    assert.equal(calls[0].args[1], hkcu)
    for (const call of calls.filter(call => call.exe === 'reg.exe')) {
      assert.equal(call.options.windowsHide, true)
      assert.equal(call.options.timeout, 1000)
      assert.equal(call.options.encoding, 'utf8')
      assert.deepEqual(Array.from(call.options.stdio), ['ignore', 'pipe', 'ignore'])
    }
  })
}

test('non-Windows keeps existing profile behavior without registry calls', async () => {
  const { bridge, calls } = await loadBridge({}, 'darwin')
  assert.equal(bridge.isWslBridgeActive(), true)
  bridge.setWslBridgeProfileState('remote', false)
  assert.equal(bridge.isWslBridgeActive('remote'), false)
  assert.deepEqual(calls, [])
})

// Exercise neighboring checked-in Vitest cases without installing the desktop
// dependency tree. Only the test registration API is adapted; bridge code runs
// unchanged, with native IO boundaries replaced above.
for (const filename of ['wsl-path-bridge.test.ts', 'wsl-path-bridge-profile.test.ts']) {
  test(`neighboring suite: ${filename}`, async t => {
    const cases = []
    const hooks = []
    const context = vm.createContext({ process: { platform: 'darwin', env: {} } })
    const vitest = new vm.SyntheticModule(['test', 'describe', 'afterEach'], function () {
      this.setExport('test', (name, fn) => cases.push({ name, fn }))
      this.setExport('describe', (_name, fn) => fn())
      this.setExport('afterEach', fn => hooks.push(fn))
    }, { context })
    const modules = new Map()
    async function linker(specifier) {
      if (specifier === 'vitest') return vitest
      if (specifier.startsWith('node:')) {
        const exports = await import(specifier)
        return new vm.SyntheticModule(Object.keys(exports), function () {
          for (const key of Object.keys(exports)) this.setExport(key, exports[key])
        }, { context })
      }
      const file = path.join(electronDir, `${specifier}.ts`)
      if (!modules.has(file)) {
        const mod = new vm.SourceTextModule(stripTypeScriptTypes(fs.readFileSync(file, 'utf8')), { context })
        modules.set(file, mod)
        await mod.link(linker)
      }
      return modules.get(file)
    }
    const suite = new vm.SourceTextModule(stripTypeScriptTypes(fs.readFileSync(path.join(electronDir, filename), 'utf8')), { context })
    await suite.link(linker)
    await suite.evaluate()
    for (const { name, fn } of cases) await t.test(name, async () => {
      try { await fn() } finally { for (const hook of hooks) await hook() }
    })
  })
}
