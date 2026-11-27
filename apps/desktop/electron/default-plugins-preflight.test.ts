import type { ChildProcess, spawn } from 'node:child_process'
import { EventEmitter } from 'node:events'
import fs from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { PassThrough } from 'node:stream'

import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { buildDesktopBackendEnv } from './backend-env'
import { desktopBackendSpawnEnv } from './backend-spawn-env'
import {
  buildDefaultPluginsPreflightInvocation,
  createPreflightGatewayRunner,
  DEFAULT_PLUGINS_LOG_PREFIX,
  DEFAULT_PLUGINS_POOL_PREFLIGHT_TIMEOUT_SECONDS,
  DEFAULT_PLUGINS_PREFLIGHT_TIMEOUT_SECONDS,
  DEFAULT_PLUGINS_TIMEOUT_SECONDS,
  parseDefaultPluginsJsonLine,
  resolveDefaultPluginsTimeoutSeconds,
  runDefaultPluginsPreflight
} from './default-plugins-preflight'
import type { InstallStamp } from './install-stamp'
import { ensureLocalGateway, runGatewayEnsure } from './local-gateway'
import { bundledPayload, payloadPythonPath, payloadTicketRuntime } from './payload-backend'
import { runPrimaryBackendStartup } from './primary-backend-startup'
import { createSourcePythonBackend } from './source-backend'

let tmpDir: string

beforeEach(async () => {
  tmpDir = await fs.realpath(await fs.mkdtemp(path.join(os.tmpdir(), 'default-plugins-preflight-')))
})

afterEach(async () => {
  await fs.rm(tmpDir, { recursive: true, force: true })
})

async function writeFakeScript(name: string, body: string): Promise<string> {
  const file = path.join(tmpDir, name)
  await fs.writeFile(file, body, { mode: 0o755 })

  return file
}

test('resolveDefaultPluginsTimeoutSeconds defaults to 120 and validates explicit overrides', () => {
  expect(DEFAULT_PLUGINS_PREFLIGHT_TIMEOUT_SECONDS).toBe(120)
  expect(DEFAULT_PLUGINS_TIMEOUT_SECONDS).toBe(120)
  expect(resolveDefaultPluginsTimeoutSeconds()).toBe(120)
  expect(resolveDefaultPluginsTimeoutSeconds(45)).toBe(45)
  expect(resolveDefaultPluginsTimeoutSeconds(12.5)).toBe(12.5)
  expect(resolveDefaultPluginsTimeoutSeconds(0)).toBe(120)
  expect(resolveDefaultPluginsTimeoutSeconds(-5)).toBe(120)
  expect(resolveDefaultPluginsTimeoutSeconds(Number.NaN)).toBe(120)
  expect(resolveDefaultPluginsTimeoutSeconds(15)).toBe(15)
})

test('parseDefaultPluginsJsonLine extracts the last JSON object line even with noisy output', () => {
  const parsed = parseDefaultPluginsJsonLine(
    'some startup warning\n{"state":"installed","pending_before":["langfuse"],"installed":["langfuse"],"owed":[],"elapsed_ms":120}\n'
  )

  expect(parsed).not.toBeNull()
  expect(parsed!.payload).toEqual({
    state: 'installed',
    pending_before: ['langfuse'],
    installed: ['langfuse'],
    owed: [],
    elapsed_ms: 120
  })
  expect(parseDefaultPluginsJsonLine('not json\n[1,2,3]\n')).toBeNull()
})

test('success (exit 0) parses JSON payload and logs a single [default-plugins] line with elapsed time', async () => {
  const payload = {
    state: 'installed',
    pending_before: ['langfuse'],
    installed: ['langfuse'],
    owed: [],
    elapsed_ms: 42
  }

  const fakeLauncher = await writeFakeScript(
    'hermes-ok',
    `#!/bin/sh\nprintf '%s\\n' '${JSON.stringify(payload)}'\nexit 0\n`
  )

  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'command', command: fakeLauncher, env: {}, shell: false },
    { cwd: tmpDir, home: tmpDir, parentEnv: { PATH: process.env.PATH }, log: line => logs.push(line) }
  )

  expect(result.skipped).toBe(false)
  expect(result.exitCode).toBe(0)
  expect(result.timedOut).toBe(false)
  expect(result.error).toBeNull()
  expect(result.payload).toEqual(payload)
  expect(logs).toHaveLength(1)
  expect(logs[0]).toMatch(new RegExp(`^\\${DEFAULT_PLUGINS_LOG_PREFIX} exit=0 \\{.*\\} elapsed_ms=\\d+$`))
  expect(logs[0]).not.toContain('\n')
  expect(logs[0]).toContain('"state":"installed"')
})

test.each([
  { code: 2, state: 'owed', owed: ['langfuse'] },
  { code: 3, state: 'timeout', owed: ['langfuse'] },
  { code: 4, state: 'skipped', owed: [] }
])('exit $code ($state) logs [default-plugins] line and never throws into startup', async ({ code, state, owed }) => {
  const payload = { state, pending_before: owed, installed: [], owed, elapsed_ms: 15 }

  const fakeLauncher = await writeFakeScript(
    `hermes-exit-${code}`,
    `#!/bin/sh\nprintf '%s\\n' '${JSON.stringify(payload)}'\nexit ${code}\n`
  )

  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'command', command: fakeLauncher, env: {}, shell: false },
    { cwd: tmpDir, home: tmpDir, parentEnv: { PATH: process.env.PATH }, log: line => logs.push(line) }
  )

  expect(result.skipped).toBe(false)
  expect(result.exitCode).toBe(code)
  expect(result.payload).toEqual(payload)
  expect(logs).toHaveLength(1)
  expect(logs[0]).toContain(`[default-plugins] exit=${code} `)
  expect(logs[0]).toContain(`"state":"${state}"`)
  expect(logs[0]).toMatch(/elapsed_ms=\d+$/)
})

test('crash (non-zero exit without JSON) logs error tail and continues', async () => {
  const fakeLauncher = await writeFakeScript(
    'hermes-crash',
    '#!/bin/sh\necho "Traceback (most recent call last):" >&2\necho "ModuleNotFoundError: No module named broken" >&2\nexit 1\n'
  )

  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'command', command: fakeLauncher, env: {}, shell: false },
    { cwd: tmpDir, home: tmpDir, parentEnv: { PATH: process.env.PATH }, log: line => logs.push(line) }
  )

  expect(result.skipped).toBe(false)
  expect(result.exitCode).toBe(1)
  expect(result.payload).toBeNull()
  expect(result.error).toContain('ModuleNotFoundError: No module named broken')
  expect(logs).toHaveLength(1)
  expect(logs[0]).toMatch(/^\[default-plugins\] exit=1 ModuleNotFoundError: No module named broken elapsed_ms=\d+$/)
})

test('spawn ENOENT and throwing log callback never throw into startup', async () => {
  const missing = path.join(tmpDir, 'does-not-exist-hermes')
  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'python', command: missing, env: {}, shell: false },
    {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: {},
      log: line => {
        logs.push(line)
        throw new Error('disk full')
      }
    }
  )

  expect(result.skipped).toBe(false)
  expect(result.exitCode).toBeNull()
  expect(result.error).toMatch(/ENOENT/i)
  expect(logs).toHaveLength(1)
  expect(logs[0]).toMatch(/^\[default-plugins\] exit=error .*elapsed_ms=\d+$/)
})

test('synchronous spawn throw is caught, logged, and never throws', async () => {
  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'python', command: '/usr/bin/python3', env: {}, shell: false },
    {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: {},
      log: line => logs.push(line),
      spawnImpl: (() => {
        throw new Error('E2BIG')
      }) as unknown as typeof spawn
    }
  )

  expect(result.exitCode).toBeNull()
  expect(result.error).toBe('E2BIG')
  expect(logs[0]).toMatch(/^\[default-plugins\] exit=error E2BIG elapsed_ms=\d+$/)
})

test('timeout kills a hanging child process and resolves cleanly', async () => {
  const pidFile = path.join(tmpDir, 'hung.pid')

  const hanger = await writeFakeScript(
    'hermes-hang',
    `#!/bin/sh\necho $$ > "${pidFile}"\nexec "${process.execPath}" -e "setInterval(() => {}, 1000)"\n`
  )

  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'command', command: hanger, env: {}, shell: false },
    {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: { PATH: process.env.PATH },
      timeoutSeconds: 0.15,
      killGraceMs: 0,
      forceKillMs: 250,
      log: line => logs.push(line)
    }
  )

  expect(result.timedOut).toBe(true)
  expect(logs).toHaveLength(1)
  expect(logs[0]).toMatch(/^\[default-plugins\] exit=timeout timed out after 0\.15s \(killed\).*elapsed_ms=\d+$/)

  // Verify the spawned child process is no longer alive.
  const pidText = await fs.readFile(pidFile, 'utf8').catch(() => '')
  const pid = Number(pidText.trim())

  if (Number.isInteger(pid) && pid > 0) {
    let alive = true

    for (let attempt = 0; attempt < 20; attempt++) {
      try {
        process.kill(pid, 0)
        await new Promise(resolve => setTimeout(resolve, 25))
      } catch {
        alive = false

        break
      }
    }

    expect(alive).toBe(false)
  }
})

test('timeout escalates SIGTERM to SIGKILL when a child ignores SIGTERM', async () => {
  const signals: (string | undefined)[] = []

  const fakeSpawn = (() => {
    const child = Object.assign(new EventEmitter(), {
      stdout: new PassThrough(),
      stderr: new PassThrough(),
      kill: (sig?: NodeJS.Signals) => {
        signals.push(sig)

        return true
      }
    }) as unknown as ChildProcess

    return child
  }) as unknown as typeof spawn

  const logs: string[] = []

  const result = await runDefaultPluginsPreflight(
    { kind: 'python', command: '/usr/bin/python3', env: {}, shell: false },
    {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: {},
      timeoutSeconds: 0.02,
      killGraceMs: 0,
      forceKillMs: 25,
      spawnImpl: fakeSpawn,
      log: line => logs.push(line)
    }
  )

  expect(result.timedOut).toBe(true)
  expect(signals).toEqual(['SIGTERM', 'SIGKILL'])
  expect(logs[0]).toMatch(/^\[default-plugins\] exit=timeout timed out after 0\.02s \(killed\) elapsed_ms=\d+$/)
})

test('remote backends and bootstrap-needed skip preflight without spawning', async () => {
  const spawnSpy = vi.fn()

  for (const backend of [
    null,
    undefined,
    { kind: 'python', command: '/usr/bin/python3', mode: 'remote', env: {}, shell: false },
    { kind: 'command', command: '/usr/bin/hermes', mode: 'ssh', env: {}, shell: false },
    { kind: 'bootstrap-needed', command: null, env: {}, shell: false }
  ]) {
    expect(buildDefaultPluginsPreflightInvocation(backend, { home: tmpDir, parentEnv: {} })).toBeNull()

    const logs: string[] = []

    const res = await runDefaultPluginsPreflight(backend, {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: {},
      log: line => logs.push(line),
      spawnImpl: spawnSpy as unknown as typeof spawn
    })

    expect(res.skipped).toBe(true)
    expect(logs).toEqual([])
  }

  expect(spawnSpy).not.toHaveBeenCalled()
})

test('primary startup ordering: remote skips preflight; local awaits preflight before gateway ensure', async () => {
  const remotePreflight = vi.fn()

  const remoteSetup = await runPrimaryBackendStartup({
    assertCurrentAttempt: () => undefined,
    connectRemote: async remote => ({ mode: 'remote', remote }),
    ensureLocalRuntime: async backend => backend,
    prepareLocalBackend: () => ({ kind: 'python', command: '/usr/bin/python3', env: {}, shell: false }),
    resolveRemote: async () => ({ baseUrl: 'https://remote.example' }),
    waitForDecision: async () => 'continue-local',
    waitForLocalStart: async () => undefined
  })

  if (remoteSetup.kind === 'local') {
    await remotePreflight()
  }

  expect(remoteSetup.kind).toBe('remote')
  expect(remotePreflight).not.toHaveBeenCalled()

  // Local path: preflight MUST finish before runGatewayEnsure executes.
  const order: string[] = []
  const localBackend = { kind: 'python', command: '/usr/bin/python3', env: {}, shell: false }

  const localSetup = await runPrimaryBackendStartup({
    assertCurrentAttempt: () => undefined,
    connectRemote: async remote => ({ mode: 'remote', remote }),
    ensureLocalRuntime: async backend => backend,
    prepareLocalBackend: () => localBackend,
    resolveRemote: async () => null,
    waitForDecision: async () => 'continue-local',
    waitForLocalStart: async () => undefined
  })

  expect(localSetup.kind).toBe('local')

  const fakeSpawn = (() => {
    const stdout = new PassThrough()

    const child = Object.assign(new EventEmitter(), {
      stdout,
      stderr: new PassThrough(),
      kill: () => true
    }) as unknown as ChildProcess

    setTimeout(() => {
      order.push('preflight-done')
      stdout.end('{"state":"nothing-pending","pending_before":[],"installed":[],"owed":[],"elapsed_ms":1}\n')
      child.emit('close', 0, null)
    }, 15)

    return child
  }) as unknown as typeof spawn

  if (localSetup.kind === 'local') {
    await runDefaultPluginsPreflight(localSetup.backend, {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: {},
      spawnImpl: fakeSpawn
    })

    await ensureLocalGateway(async () => {
      order.push('gateway-ensure')

      return {
        code: 0,
        stdout: JSON.stringify({
          state: 'ready',
          endpoint: {
            profile_id: tmpDir,
            instance_id: 'inst-1',
            authority_epoch: 1,
            runtime_protocol: 1,
            api_origin: 'http://127.0.0.1:8642',
            capabilities: ['session-authority-v1'],
            supervisor: 'detached'
          }
        })
      }
    })
  }

  expect(order).toEqual(['preflight-done', 'gateway-ensure'])
})

test('createPreflightGatewayRunner runs preflight once per token and orders hooks before ensure', async () => {
  const order: string[] = []
  const controller = new AbortController()

  const runner = createPreflightGatewayRunner({
    hermesHome: tmpDir,
    resolveCwd: () => tmpDir,
    buildSpawnEnv: env => ({ ...env, SPAWN_ENV: '1' }),
    buildParentEnv: profile => ({ PROFILE: profile }),
    runPreflight: async () => {
      order.push('preflight')

      return {
        skipped: false,
        exitCode: 0,
        timedOut: false,
        aborted: false,
        elapsedMs: 1,
        payload: null,
        error: null,
        logLine: null
      }
    },
    runEnsure: async (_backend, _cwd, _home, parentEnv) => {
      order.push(`ensure:${parentEnv.PROFILE}`)

      return 'ok'
    },
    log: () => undefined,
    getSignal: () => controller.signal
  })

  const backend = { kind: 'python', command: '/usr/bin/python3', env: {}, shell: false }
  const once = { ran: false }

  await runner.run(backend, 'default', {
    once,
    onBefore: () => {
      order.push('before')
    },
    onAfter: () => {
      order.push('after')
    }
  })
  await runner.run(backend, 'default', {
    once,
    onBefore: () => {
      order.push('before-2')
    },
    onAfter: () => {
      order.push('after-2')
    }
  })

  expect(order).toEqual(['before', 'preflight', 'after', 'ensure:default', 'ensure:default'])
})

test('abort signal terminates an in-flight preflight child and settles cleanly', async () => {
  const controller = new AbortController()
  const signals: (string | undefined)[] = []

  const fakeSpawn = (() => {
    const child = Object.assign(new EventEmitter(), {
      stdout: new PassThrough(),
      stderr: new PassThrough(),
      kill: (sig?: NodeJS.Signals) => {
        signals.push(sig)
        setTimeout(() => child.emit('close', null, sig ?? 'SIGTERM'), 5)

        return true
      }
    }) as unknown as ChildProcess

    return child
  }) as unknown as typeof spawn

  const logs: string[] = []

  const promise = runDefaultPluginsPreflight(
    { kind: 'python', command: '/usr/bin/python3', env: {}, shell: false },
    {
      cwd: tmpDir,
      home: tmpDir,
      parentEnv: {},
      signal: controller.signal,
      timeoutSeconds: 30,
      spawnImpl: fakeSpawn,
      log: line => logs.push(line)
    }
  )

  controller.abort()
  const result = await promise

  expect(result.aborted).toBe(true)
  expect(signals).toEqual(['SIGTERM'])
  expect(logs[0]).toMatch(/^\[default-plugins\] exit=aborted aborted while starting elapsed_ms=\d+$/)
})

test('bundled payload uses storePython + ticketEnv and keeps backend env parity', () => {
  const stamp = {
    schemaVersion: 1,
    commit: 'a'.repeat(40),
    commitDate: null,
    branch: null,
    builtAt: null,
    dirty: false,
    source: 'ci',
    distribution: 'desktop-app',
    updateMechanism: 'electron-updater',
    baseVersion: null,
    displayVersion: null,
    distance: null,
    payload: 'bundled',
    tag: 'v1.0.0',
    runtime: {
      repoDir: 'app',
      toolsDir: 'tools',
      storePython: 'tools/python/bin/python3',
      sitePackages: 'venv/site-packages',
      commands: { hermes: 'bin/hermes' }
    }
  } as unknown as InstallStamp

  const resources = path.join(tmpDir, 'resources')
  const payload = bundledPayload(resources, stamp)!
  const userHome = path.join(tmpDir, 'user')

  const backend = {
    kind: 'python',
    command: payload.shim,
    env: desktopBackendSpawnEnv({
      ...buildDesktopBackendEnv({ currentEnv: { PATH: '/custom/bin' }, platform: 'linux', homedir: userHome }),
      HERMES_RUNTIME_DIR: payload.toolsDir
    }),
    ...payloadTicketRuntime(payload, { HOME: userHome }, 'linux'),
    shell: false
  }

  const inv = buildDefaultPluginsPreflightInvocation(backend, {
    home: path.join(tmpDir, 'hermes-home'),
    parentEnv: {
      HOME: userHome,
      HERMES_DESKTOP_DEV_SERVER: 'http://127.0.0.1:5174'
    },
    timeoutSeconds: 180,
    isWindows: false
  })!

  expect(inv.command).toBe(payload.storePython)
  expect(inv.shell).toBe(false)
  expect(inv.args).toEqual(['-P', '-m', 'hermes_cli.default_plugins', 'ensure', '--json', '--timeout', '180'])
  expect(inv.env.HERMES_HOME).toBe(path.join(tmpDir, 'hermes-home'))
  expect(inv.env.HERMES_RUNTIME_DIR).toBe(payload.toolsDir)
  expect(inv.env.PYTHONPATH).toBe(payloadPythonPath(payload))
  expect(inv.env.PYTHONPYCACHEPREFIX).toBe(path.posix.join(userHome, '.cache', 'hermes-pycache'))
  expect(inv.env.GATEWAY_ON_ALL_ADAPTERS_DOWN).toBe('stay_alive')
  expect(inv.env.PATH).toContain('/custom/bin')
  expect(inv.env.HERMES_DESKTOP_DEV_SERVER).toBeUndefined()
})

test('source checkout backend uses its python interpreter and exact PYTHONPATH/env', () => {
  const root = path.join(tmpDir, 'checkout')

  const source = createSourcePythonBackend(root, '/opt/python/bin/python3', ['gateway', 'ensure', '--json'], {
    isWindows: false,
    env: { PATH: '/dev/bin' }
  })!

  const spawnBackend = { ...source, env: desktopBackendSpawnEnv(source.env) }

  const inv = buildDefaultPluginsPreflightInvocation(spawnBackend, {
    home: path.join(tmpDir, 'home'),
    parentEnv: { CUSTOM_FLAG: 'yes', HERMES_DESKTOP_DEV_SERVER: 'http://127.0.0.1:5174' },
    isWindows: false
  })!

  expect(inv.command).toBe('/opt/python/bin/python3')
  expect(inv.shell).toBe(false)
  expect(inv.args).toEqual(['-P', '-m', 'hermes_cli.default_plugins', 'ensure', '--json', '--timeout', '120'])
  expect(inv.env.HERMES_HOME).toBe(path.join(tmpDir, 'home'))
  expect(inv.env.PYTHONPATH).toBe(root)
  expect(inv.env.PATH).toContain('/dev/bin')
  expect(inv.env.CUSTOM_FLAG).toBe('yes')
  expect(inv.env.GATEWAY_ON_ALL_ADAPTERS_DOWN).toBe('stay_alive')
  expect(inv.env.HERMES_DESKTOP_DEV_SERVER).toBeUndefined()
})

test('installed launcher backend uses --run-module and quotes Windows .cmd commands', () => {
  const cmdPath = 'C:\\Users\\John Doe\\AppData\\Local\\hermes\\bin\\hermes.cmd'

  const inv = buildDefaultPluginsPreflightInvocation(
    { kind: 'command', command: cmdPath, shell: true, env: { HERMES_RUNTIME_DIR: 'C:\\tools' } },
    { home: 'C:\\Users\\John Doe\\.hermes', parentEnv: {}, isWindows: true }
  )!

  expect(inv.command).toBe(`"${cmdPath}"`)
  expect(inv.shell).toBe(true)
  expect(inv.args).toEqual([
    '--run-module',
    'hermes_cli.default_plugins',
    'ensure',
    '--json',
    '--timeout',
    '120'
  ])
  expect(inv.env.HERMES_HOME).toBe('C:\\Users\\John Doe\\.hermes')
  expect(inv.env.HERMES_RUNTIME_DIR).toBe('C:\\tools')
})

test('preflight child observes the exact same env as runGatewayEnsure', async () => {
  const dumpBin = await writeFakeScript(
    'dump-env',
    `#!${process.execPath}
const out = {
  argv: process.argv.slice(2),
  HERMES_HOME: process.env.HERMES_HOME,
  HERMES_RUNTIME_DIR: process.env.HERMES_RUNTIME_DIR,
  PATH: process.env.PATH,
  PYTHONUTF8: process.env.PYTHONUTF8,
  GATEWAY_ON_ALL_ADAPTERS_DOWN: process.env.GATEWAY_ON_ALL_ADAPTERS_DOWN,
  HERMES_DESKTOP_DEV_SERVER: process.env.HERMES_DESKTOP_DEV_SERVER ?? null,
  CUSTOM_VAR: process.env.CUSTOM_VAR
}
console.log(JSON.stringify(out))
`
  )

  const home = path.join(tmpDir, 'shared-home')

  const backendEnv = desktopBackendSpawnEnv({
    ...buildDesktopBackendEnv({ currentEnv: { PATH: process.env.PATH } }),
    HERMES_RUNTIME_DIR: path.join(tmpDir, 'tools')
  })

  const parentEnv = {
    PATH: process.env.PATH,
    CUSTOM_VAR: 'preserved',
    HERMES_DESKTOP_DEV_SERVER: 'http://127.0.0.1:5174',
    HERMES_HOME: '/wrong-inherited-home'
  }

  const backend = { kind: 'command', command: dumpBin, args: ['gateway', 'ensure', '--json'], env: backendEnv, shell: false }

  const preflight = await runDefaultPluginsPreflight(backend, { cwd: tmpDir, home, parentEnv })
  const gateway = await runGatewayEnsure(backend, tmpDir, home, parentEnv)
  const gatewayDump = JSON.parse(gateway.stdout)

  expect(preflight.exitCode).toBe(0)
  expect(preflight.payload).not.toBeNull()

  const { argv: preflightArgv, ...preflightEnv } = preflight.payload as Record<string, unknown>
  const { argv: gatewayArgv, ...gatewayEnv } = gatewayDump as Record<string, unknown>

  expect(preflightArgv).toEqual(['--run-module', 'hermes_cli.default_plugins', 'ensure', '--json', '--timeout', '120'])
  expect(gatewayArgv).toEqual(['gateway', 'ensure', '--json'])
  expect(preflightEnv).toEqual(gatewayEnv)
})

test('createPreflightGatewayRunner aborts cleanly without calling runEnsure when signal is aborted', async () => {
  const alreadyAborted = new AbortController()
  alreadyAborted.abort(new Error('lifecycle stopped'))

  const runEnsure = vi.fn(async () => ({ code: 0, stdout: '{}', stderr: '' }))
  const runPreflight = vi.fn(runDefaultPluginsPreflight)
  const onBefore = vi.fn()
  const onAfter = vi.fn()

  const runner = createPreflightGatewayRunner({
    hermesHome: tmpDir,
    resolveCwd: () => tmpDir,
    buildSpawnEnv: env => env,
    buildParentEnv: () => ({}),
    runEnsure,
    log: () => undefined,
    getSignal: () => alreadyAborted.signal,
    runPreflight
  })

  const backend = { kind: 'python', command: process.execPath, env: {}, shell: false }

  await expect(runner.run(backend, 'default', { onBefore, onAfter })).rejects.toThrow('lifecycle stopped')
  expect(onBefore).not.toHaveBeenCalled()
  expect(runPreflight).not.toHaveBeenCalled()
  expect(onAfter).not.toHaveBeenCalled()
  expect(runEnsure).not.toHaveBeenCalled()

  // Also abort mid-preflight: runPreflight returns aborted=true and runEnsure must still NOT be called.
  const midFlight = new AbortController()

  const midRunner = createPreflightGatewayRunner({
    hermesHome: tmpDir,
    resolveCwd: () => tmpDir,
    buildSpawnEnv: env => env,
    buildParentEnv: () => ({}),
    runEnsure,
    log: () => undefined,
    getSignal: () => midFlight.signal,
    runPreflight: async () => {
      midFlight.abort(new Error('aborted during preflight'))

      return {
        skipped: false,
        exitCode: null,
        timedOut: false,
        aborted: true,
        elapsedMs: 1,
        payload: null,
        error: 'aborted while starting',
        logLine: null
      }
    }
  })

  await expect(midRunner.run(backend, 'default', { onBefore, onAfter })).rejects.toThrow('aborted during preflight')
  expect(onAfter).not.toHaveBeenCalled()
  expect(runEnsure).not.toHaveBeenCalled()
})

test('pooled local dial preflight budget fits inside renderer dial timeouts and forwards timeoutSeconds', async () => {
  const poolTimeoutMs = DEFAULT_PLUGINS_POOL_PREFLIGHT_TIMEOUT_SECONDS * 1000
  const killGraceMs = Math.min(10_000, Math.max(3_000, Math.round(poolTimeoutMs * 0.1)))
  const forceKillMs = 1_000
  const maxPoolPreflightMs = poolTimeoutMs + killGraceMs + forceKillMs
  const gatewayEnsureMaxMs = 75_000 // GATEWAY_ENSURE_PROTOCOL_DEADLINE_MS (60s) + grace (15s)
  const sourceSwitchDialTimeoutMs = 135_000 // SOURCE_SWITCH_DIAL_TIMEOUT_MS
  const legacyPoolDialTimeoutMs = 20_000 // RECONNECT_ATTEMPT_TIMEOUT_MS / DESCRIPTOR_LOOKUP_TIMEOUT_MS

  expect(maxPoolPreflightMs + gatewayEnsureMaxMs).toBeLessThan(sourceSwitchDialTimeoutMs)
  expect(maxPoolPreflightMs).toBeLessThan(legacyPoolDialTimeoutMs)

  const seenTimeouts: Array<number | undefined> = []
  const runEnsure = vi.fn(async () => ({ code: 0, stdout: '{}', stderr: '' }))

  const runner = createPreflightGatewayRunner({
    hermesHome: tmpDir,
    resolveCwd: () => tmpDir,
    buildSpawnEnv: env => env,
    buildParentEnv: () => ({}),
    runEnsure,
    log: () => undefined,
    getSignal: () => new AbortController().signal,
    runPreflight: async (_backend, opts) => {
      seenTimeouts.push(opts.timeoutSeconds)

      return {
        skipped: false,
        exitCode: 0,
        timedOut: false,
        aborted: false,
        elapsedMs: 1,
        payload: { state: 'nothing-pending' },
        error: null,
        logLine: null
      }
    }
  })

  const backend = { kind: 'python', command: process.execPath, env: {}, shell: false }
  await runner.run(backend, 'work', { timeoutSeconds: DEFAULT_PLUGINS_POOL_PREFLIGHT_TIMEOUT_SECONDS })

  expect(seenTimeouts).toEqual([DEFAULT_PLUGINS_POOL_PREFLIGHT_TIMEOUT_SECONDS])
  expect(runEnsure).toHaveBeenCalledTimes(1)
})
