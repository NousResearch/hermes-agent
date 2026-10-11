import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import vm from 'node:vm'

import ts from 'typescript'
import { describe, expect, it, vi } from 'vitest'

import * as linux from './linux-gpu-fallback'
import * as nvidia from './linux-nvidia-egl-fallback'
import * as sandbox from './windows-sandbox-fallback'
import * as windows from './windows-stack-cookie-fallback'

// Execute the actual main.ts pre-ready prologue, including the hoisted lock
// function and the secondary's app.exit path. Only native Electron/OS/IO seams
// are substituted; marker parsing, serialization and recovery decisions are real.
const source = fs.readFileSync(new URL('./main.ts', import.meta.url), 'utf8')
const ast = ts.createSourceFile('main.ts', source, ts.ScriptTarget.Latest, true)

const end = ast.statements.findIndex(
  statement =>
    ts.isVariableStatement(statement) &&
    statement.declarationList.declarations.some(declaration => declaration.name.getText(ast) === 'HERMES_HOME')
)

if (end < 0) {
  throw new Error('Missing pre-ready startup boundary')
}

const prologue = ast.statements
  .slice(0, end)
  .filter(statement => !ts.isImportDeclaration(statement))
  .map(statement => statement.getText(ast))
  .join('\n')

const executable = ts.transpileModule(prologue, {
  compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None }
}).outputText

const userData = '/isolated-desktop-user-data'

const markerPaths = [
  linux.linuxGpuMarkerPath(userData),
  nvidia.nvidiaEglMarkerPath(userData),
  sandbox.sandboxMarkerPath(userData),
  windows.gpuStackCookieMarkerPath(userData)
]

const hosts = [
  { name: 'Linux Mesa', platform: 'linux', nvidia: false, markers: [markerPaths[0], markerPaths[2]] },
  { name: 'Linux NVIDIA', platform: 'linux', nvidia: true, markers: markerPaths.slice(0, 3) },
  { name: 'Windows', platform: 'win32', nvidia: false, markers: markerPaths.slice(2) },
  { name: 'macOS', platform: 'darwin', nvidia: false, markers: [] }
] as const

type Host = (typeof hosts)[number]

function launch(host: Host, files: Map<string, string>, locks = [true], stalePid: number | null = null) {
  const events: string[] = []
  const listeners: string[] = []
  const switches: string[] = []
  const exit = new Error('native app.exit')

  const readFileSync = ((file: fs.PathOrFileDescriptor) => {
    const value = files.get(String(file))

    if (value === undefined) {
      throw new Error('ENOENT')
    }

    return value
  }) as typeof fs.readFileSync

  const io = {
    readFileSync,
    mkdirSync: vi.fn() as typeof fs.mkdirSync,
    writeFileSync: ((file: fs.PathOrFileDescriptor, value: string) => {
      events.push(`write:${String(file)}`)
      files.set(String(file), String(value))
    }) as typeof fs.writeFileSync
  }

  const context = {
    ...linux,
    ...nvidia,
    ...sandbox,
    ...windows,
    // These helpers default to the real process.platform. Feed them the same
    // simulated native platform seen by the executing production prologue.
    decideLinuxGpuLaunch: (options: Parameters<typeof linux.decideLinuxGpuLaunch>[0]) =>
      linux.decideLinuxGpuLaunch({ ...options, platform: host.platform }),
    decideWindowsSandboxLaunch: (options: Parameters<typeof sandbox.decideWindowsSandboxLaunch>[0]) =>
      sandbox.decideWindowsSandboxLaunch({ ...options, platform: host.platform }),
    readLinuxGpuMarker: (dir: string) => linux.readLinuxGpuMarker(dir, io),
    writeLinuxGpuMarker: (dir: string, marker: linux.LinuxGpuMarker) => linux.writeLinuxGpuMarker(dir, marker, io),
    readNvidiaEglMarker: (dir: string) => nvidia.readNvidiaEglMarker(dir, io),
    writeNvidiaEglMarker: (dir: string, marker: nvidia.NvidiaEglMarker) => nvidia.writeNvidiaEglMarker(dir, marker, io),
    readSandboxMarker: (dir: string) => sandbox.readSandboxMarker(dir, io),
    writeSandboxMarker: (dir: string, marker: sandbox.SandboxMarker) => sandbox.writeSandboxMarker(dir, marker, io),
    readGpuStackCookieMarker: (dir: string) => windows.readGpuStackCookieMarker(dir, io),
    writeGpuStackCookieMarker: (dir: string, marker: windows.GpuStackCookieMarker) =>
      windows.writeGpuStackCookieMarker(dir, marker, io),
    app: {
      isPackaged: false,
      getPath: () => userData,
      getAppPath: () => '/isolated-app',
      getVersion: () => '0.0.0',
      requestSingleInstanceLock: () => {
        events.push('lock')

        return locks.shift() ?? false
      },
      disableHardwareAcceleration: () => switches.push('disable-hardware-acceleration'),
      commandLine: { appendSwitch: (name: string) => switches.push(name) },
      on: (event: string) => listeners.push(event),
      exit: (code: number) => {
        events.push(`exit:${code}`)
        throw exit // Native app.exit does not run before-quit cleanup.
      }
    },
    removeStaleSingletonLock: (dir: string) => {
      expect(dir).toBe(userData)
      events.push('stale-lock-probe')

      return stalePid
    },
    fs: {
      readFileSync: () => (host.nvidia ? 'NVRM version: NVIDIA UNIX x86_64 Kernel Module 580.82.09' : ''),
      existsSync: () => false
    },
    os,
    path,
    process: {
      platform: host.platform,
      argv: ['electron', '/isolated-app'],
      env: {},
      execPath: '/isolated-app/electron'
    },
    console: { log: vi.fn(), warn: vi.fn(), error: vi.fn() },
    ipcMain: { handle: vi.fn() },
    applyDesktopIdentity: () => null,
    isWslEnvironment: () => false,
    glassSupportedOn: () => false,
    translucencySupportedOn: () => false,
    detectRemoteDisplay: () => null,
    resolveDevCdpPort: () => ({ port: null }),
    describeDevCdpDecision: () => null,
    resolveLinuxPasswordStore: () => ({}),
    grantAllApplicationPackagesAcl: () => ({ ok: false }),
    INSTALL_STAMP: null
  }

  let exited = false

  try {
    vm.runInNewContext(executable, context, { timeout: 1000 })
  } catch (error) {
    if (error !== exit) {
      throw error
    }

    exited = true
  }

  return { events, listeners, switches, exited }
}

describe('production startup marker ownership', () => {
  for (const host of hosts) {
    it(`${host.name}: acquires ownership before writing any pre-ready marker`, () => {
      const files = new Map<string, string>()
      const result = launch(host, files)
      expect(result.exited).toBe(false)
      expect(result.events[0]).toBe('lock')
      expect([...files.keys()].sort()).toEqual([...host.markers].sort())

      for (const bytes of files.values()) {
        expect(JSON.parse(bytes).state).toBe('booting')
      }
    })

    for (const state of ['ok', 'booting', 'fallback'] as const) {
      it(`${host.name}: repeated lock losers preserve ${state} marker bytes and register no recovery listeners`, () => {
        const files = new Map(markerPaths.map(file => [file, `${JSON.stringify({ state, bootAborts: 1 })}\n`]))
        const before = [...files.entries()]

        for (let attempt = 0; attempt < 3; attempt++) {
          const result = launch(host, files, [false])
          expect(result.exited).toBe(true)
          expect(result.events).toEqual(['lock', 'stale-lock-probe', 'exit:0'])
          expect(result.listeners).toEqual([])
          expect([...files.entries()]).toEqual(before)
        }

        if (state === 'ok') {
          const nextPrimary = launch(host, files)
          expect(nextPrimary.exited).toBe(false)
          expect(nextPrimary.switches).not.toContain('use-angle')
          expect(nextPrimary.switches).not.toContain('no-sandbox')
          expect(nextPrimary.switches).not.toContain('disable-hardware-acceleration')
        }
      })
    }
  }

  it('a Linux secondary does not create absent markers', () => {
    const files = new Map<string, string>()
    expect(launch(hosts[1], files, [false]).exited).toBe(true)
    expect(files.size).toBe(0)
  })

  it('a stale-lock retry winner writes only after the second acquisition', () => {
    const files = new Map<string, string>()
    const result = launch(hosts[1], files, [false, true], 99999999)
    expect(result.exited).toBe(false)
    expect(result.events.slice(0, 3)).toEqual(['lock', 'stale-lock-probe', 'lock'])
    expect(result.events.slice(3).sort()).toEqual(hosts[1].markers.map(file => `write:${file}`).sort())
    expect(result.listeners).toEqual(['child-process-gone', 'child-process-gone'])
  })

  it('a stale-lock retry loser still writes nothing', () => {
    const files = new Map<string, string>()
    const result = launch(hosts[1], files, [false, false], 99999999)
    expect(result.events).toEqual(['lock', 'stale-lock-probe', 'lock', 'exit:0'])
    expect(result.listeners).toEqual([])
    expect(files.size).toBe(0)
  })
})
