import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { createDesktopShellRuntime, type DesktopShellRuntimeDependencies } from './desktop-shell-runtime'

function fixture(overrides: Record<string, unknown> = {}) {
  const calls = {
    about: [] as unknown[],
    logs: [] as string[],
    scripts: [] as Array<{ filename: string; source: string }>,
    spawned: [] as Array<{ command: string; args: string[] }>,
    quit: 0,
    release: [] as string[]
  }
  const app = {
    getVersion: () => '0.17.6',
    getLocale: () => 'en-US',
    getPath: (name: string) => `/tmp/${name}`,
    getGPUInfo: async () => ({ gpuDevice: [{ vendorId: 0x10de }] }),
    setAboutPanelOptions: (value: unknown) => calls.about.push(value),
    showAboutPanel: vi.fn(),
    relaunch: vi.fn(),
    quit: () => {
      calls.quit += 1
    }
  }
  const deps = {
    ACTIVE_HERMES_ROOT: '/agent',
    APP_NAME: 'Hermes',
    HERMES_HOME: '/home',
    INSTALL_STAMP: null,
    IS_PACKAGED: false,
    IS_WINDOWS: false,
    VENV_ROOT: '/agent/venv',
    app,
    buildNoSandboxRelaunchArgs: (args: string[]) => args,
    exitAfterBackendShutdown: vi.fn(async () => undefined),
    fileExists: () => false,
    findSystemPython: async () => '/usr/bin/python3',
    fs: {
      readFileSync: () => '__version__ = "1.2.3"',
      statSync: () => ({ birthtimeMs: Date.now() - 5 * 86_400_000 }),
      writeFileSync: (filename: string, source: string) => calls.scripts.push({ filename, source })
    },
    getVenvPython: () => '/agent/venv/bin/python',
    hiddenWindowsChildOptions: (options: unknown) => options,
    isHermesSourceRoot: () => true,
    loadInstallStamp: () => null,
    os: { homedir: () => '/home', release: () => '6.0', userInfo: () => ({ username: 'axl' }) },
    path,
    process: {
      arch: 'arm64',
      argv: ['Hermes'],
      env: {},
      execPath: '/usr/bin/electron',
      pid: 412,
      platform: 'linux',
      versions: { electron: '40.0', node: '24.0' }
    },
    releaseBackendLock: async (root: string, tag: string) => {
      calls.release.push(`${root}:${tag}`)
    },
    rememberLog: (line: string) => calls.logs.push(line),
    resolveUpdateRoot: () => '/agent',
    runGit: async () => ({ code: 0, stdout: '', stderr: '' }),
    setQuittingForHandoff: vi.fn(),
    spawn: (command: string, args: string[]) => {
      calls.spawned.push({ command, args })
      return { unref: vi.fn() }
    }
  }

  return {
    calls,
    deps: { ...deps, ...overrides } as unknown as DesktopShellRuntimeDependencies,
    app
  }
}

afterEach(() => vi.useRealTimers())

test('early version resolution reads Hermes source and falls back to the Electron app version', () => {
  const source = fixture({ fileExists: () => true })
  assert.equal(createDesktopShellRuntime(source.deps).resolveHermesVersion(), '1.2.3')

  const missing = fixture()
  assert.equal(createDesktopShellRuntime(missing.deps).resolveHermesVersion(), '0.17.6')
})

test('About refreshes the version just before opening the native panel', async () => {
  const f = fixture({ fileExists: () => true })
  const runtime = createDesktopShellRuntime(f.deps)
  runtime.showAboutPanelFresh()
  await vi.waitFor(() => assert.equal(f.calls.about.length, 1))
  assert.deepEqual(f.calls.about[0], {
    applicationName: 'Hermes',
    applicationVersion: '1.2.3',
    copyright: 'Copyright © 2026 Nous Research'
  })
  assert.equal(f.app.showAboutPanel.mock.calls.length, 1)
})

test('version info reports a proved on-disk bundle swap while relaunch stays injected', async () => {
  const f = fixture({
    INSTALL_STAMP: { commit: 'a'.repeat(40), builtAt: 'old' },
    IS_PACKAGED: true,
    loadInstallStamp: () => ({ commit: 'a'.repeat(40), builtAt: 'new' }),
    runGit: async () => ({ code: 1, stdout: '', stderr: 'unavailable' })
  })
  const runtime = createDesktopShellRuntime(f.deps)
  const version = await runtime.getVersionInfo()
  assert.equal(version.appVersion, '0.17.6')
  assert.equal(version.bundleSwapPending, true)
  assert.equal(version.bundleOutOfSync, false)

  await runtime.relaunchAfterBundleSwap()
  assert.equal(f.app.relaunch.mock.calls.length, 1)
  assert.equal(vi.mocked(f.deps.exitAfterBackendShutdown).mock.calls.length, 1)
  assert.match(f.calls.logs[0], /renderer requested an app relaunch/)
})

test('machine profile preserves account age, locale, board model, and Chromium GPU facts', async () => {
  vi.useFakeTimers()
  vi.setSystemTime(new Date('2026-09-23T00:00:00Z'))
  const f = fixture({
    fs: {
      readFileSync: () => 'NVIDIA_DGX_Spark\0',
      statSync: () => ({ birthtimeMs: Date.now() - 5 * 86_400_000 })
    }
  })
  const profile = await createDesktopShellRuntime(f.deps).getMachineProfile()
  assert.deepEqual(profile, {
    ageDays: 5,
    arch: 'arm64',
    locale: 'en-US',
    model: 'NVIDIA_DGX_Spark',
    nvidia: true,
    platform: 'linux',
    release: '6.0',
    username: 'axl'
  })
})

test('unknown uninstall mode and absent agent refuse without writes, spawn, or quit', async () => {
  const f = fixture()
  const runtime = createDesktopShellRuntime(f.deps)
  const invalid = await runtime.runDesktopUninstall('wrong')
  const absent = await runtime.runDesktopUninstall('full')
  assert.equal('error' in invalid && invalid.error, 'invalid-mode')
  assert.equal('error' in absent && absent.error, 'agent-missing')
  assert.deepEqual(f.calls.scripts, [])
  assert.deepEqual(f.calls.spawned, [])
  assert.equal(f.calls.quit, 0)
  assert.deepEqual(f.calls.release, [])
})

test('summary probe consumes the final JSON line and adds the running bundle path', async () => {
  vi.useFakeTimers()
  const child = new EventEmitter() as EventEmitter & { stdout: EventEmitter }
  child.stdout = new EventEmitter()
  const f = fixture({
    fileExists: () => true,
    process: {
      arch: 'x64',
      argv: ['Hermes'],
      env: {},
      execPath: '/Applications/Hermes.app/Contents/MacOS/Hermes',
      pid: 412,
      platform: 'darwin',
      versions: { electron: '40.0', node: '24.0' }
    },
    spawn: () => child
  })
  const pending = createDesktopShellRuntime(f.deps).getUninstallSummary()
  child.stdout.emit('data', Buffer.from('diagnostic\n{"agent_installed":true}\n'))
  child.emit('exit', 0)
  const summary = (await pending) as { agent_installed: boolean; running_app_path: string }
  assert.equal(summary.agent_installed, true)
  assert.equal(summary.running_app_path, '/Applications/Hermes.app')
  assert.equal(f.calls.quit, 0)
  assert.deepEqual(f.calls.scripts, [])
})

test('script write failure refuses before spawning a detached remover', async () => {
  const f = fixture({
    fileExists: () => true,
    fs: {
      writeFileSync: () => {
        throw new Error('disk full')
      }
    }
  })
  const result = await createDesktopShellRuntime(f.deps).runDesktopUninstall('gui')
  assert.equal('error' in result && result.error, 'script-write-failed')
  assert.deepEqual(f.calls.spawned, [])
  assert.equal(f.calls.quit, 0)
})

test('valid uninstall uses injected script and detached child, then sets handoff before delayed quit', async () => {
  vi.useFakeTimers()
  const f = fixture({
    fileExists: () => true,
    findSystemPython: async () => 'C:\\Python313\\python.exe',
    IS_WINDOWS: true,
    process: {
      arch: 'x64',
      argv: ['Hermes.exe'],
      env: {},
      execPath: 'C:\\Users\\axl\\AppData\\Local\\Programs\\Hermes\\Hermes.exe',
      pid: 412,
      platform: 'win32',
      versions: { electron: '40.0', node: '24.0' }
    }
  })
  const runtime = createDesktopShellRuntime(f.deps)
  const result = await runtime.runDesktopUninstall('full')
  assert.equal(result.ok, true)
  assert.deepEqual(f.calls.release, ['/agent:uninstall'])
  assert.equal(f.calls.scripts.length, 1)
  assert.match(f.calls.scripts[0].filename, /hermes-uninstall-\d+\.cmd$/)
  assert.match(f.calls.scripts[0].source, /Python313/)
  assert.equal(f.calls.spawned.length, 1)
  assert.equal(f.calls.quit, 0)
  assert.equal(vi.mocked(f.deps.setQuittingForHandoff).mock.calls.length, 1)
  await vi.advanceTimersByTimeAsync(800)
  assert.equal(f.calls.quit, 1)
})
