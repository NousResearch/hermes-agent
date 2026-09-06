import assert from 'node:assert/strict'
import path from 'node:path'

import { describe, it } from 'vitest'

import {
  isRelaunchableDesktopPluginHost,
  parseTerminatedPluginServiceHost,
  type PluginHostRestoreDeps,
  recordStoppedDesktopPluginHost,
  restoreStoppedDesktopPluginHosts,
  STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION,
  stoppedPluginHostsPath
} from './desktop-plugin-host-restore'

const HOME = 'C:\\Users\\u\\AppData\\Local\\hermes'
const VBS = path.join(HOME, 'desktop-plugins', 'llm-usage-tracker', 'service-host.vbs')
const WSCRIPT = 'C:\\Windows\\system32\\wscript.exe'

function host(overrides: Partial<{ pid: number; createdAt: number; argv: string[]; cwd: string | null }> = {}) {
  return { pid: 24692, createdAt: 1788436393.1, argv: [WSCRIPT, VBS], cwd: 'C:\\Users\\u', ...overrides }
}

function memoryFs(existing: Set<string>, files: Map<string, string>) {
  const spawned: Array<[string, string[], { cwd?: string }]> = []
  const log: string[] = []

  const deps: PluginHostRestoreDeps = {
    isWindows: true,
    now: () => 1_000,
    existsSync: target => existing.has(target) || files.has(target),
    readFileSync: target => {
      const contents = files.get(target)

      if (contents === undefined) {throw new Error('ENOENT')}

      return contents
    },
    writeFileSync: (target, contents) => {
      files.set(target, contents)
    },
    mkdirSync: () => {},
    rmSync: target => {
      files.delete(target)
    },
    spawn: (command, args, options) => {
      spawned.push([command, args, options])
    },
    log: line => {
      log.push(line)
    }
  }

  return { deps, spawned, log }
}

describe('parseTerminatedPluginServiceHost', () => {
  it('accepts the scanner shape and rejects anything else', () => {
    assert.deepEqual(
      parseTerminatedPluginServiceHost({ pid: 5, created_at: 90.5, argv: ['wscript.exe', 'x.vbs'], cwd: null }),
      { pid: 5, createdAt: 90.5, argv: ['wscript.exe', 'x.vbs'], cwd: null }
    )
    assert.equal(parseTerminatedPluginServiceHost({ pid: 0, created_at: 90.5, argv: ['a'], cwd: null }), null)
    assert.equal(parseTerminatedPluginServiceHost({ pid: 5, created_at: -1, argv: ['a'], cwd: null }), null)
    assert.equal(parseTerminatedPluginServiceHost({ pid: 5, created_at: 1, argv: [], cwd: null }), null)
    assert.equal(parseTerminatedPluginServiceHost({ pid: 5, created_at: 1, argv: ['a'], cwd: 7 }), null)
    assert.equal(parseTerminatedPluginServiceHost('nope'), null)
  })
})

describe('isRelaunchableDesktopPluginHost', () => {
  it('accepts only a Windows Script Host running a .vbs under desktop-plugins that still exists', () => {
    const exists = (target: string) => target === VBS

    assert.equal(isRelaunchableDesktopPluginHost(HOME, host(), exists), true)
    assert.equal(isRelaunchableDesktopPluginHost(HOME, host({ argv: ['C:\\Windows\\system32\\cscript.exe', `"${VBS}"`] }), exists), true)
    assert.equal(isRelaunchableDesktopPluginHost(HOME, host({ argv: ['C:\\Python\\python.exe', VBS] }), exists), false)
    assert.equal(
      isRelaunchableDesktopPluginHost(HOME, host({ argv: [WSCRIPT, path.join(HOME, 'desktop-plugins', 'x', 'run.cmd')] }), exists),
      false
    )
    assert.equal(isRelaunchableDesktopPluginHost(HOME, host({ argv: [WSCRIPT, 'C:\\elsewhere\\service-host.vbs'] }), exists), false)
    assert.equal(isRelaunchableDesktopPluginHost(HOME, host(), () => false), false)
  })
})

describe('record + restore', () => {
  it('records a stopped supervisor once per script and relaunches it detached, then clears the ledger', () => {
    const files = new Map<string, string>()
    const { deps, spawned, log } = memoryFs(new Set([VBS]), files)

    assert.equal(recordStoppedDesktopPluginHost(HOME, host(), deps), true)
    assert.equal(recordStoppedDesktopPluginHost(HOME, host({ pid: 24700 }), deps), true)

    const ledger = JSON.parse(files.get(stoppedPluginHostsPath(HOME))!)
    assert.equal(ledger.schemaVersion, STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION)
    assert.equal(ledger.hosts.length, 1)
    assert.equal(ledger.hosts[0].pid, 24700)
    assert.equal(ledger.hosts[0].stoppedAt, 1_000)

    const outcome = restoreStoppedDesktopPluginHosts(HOME, deps)

    assert.deepEqual(outcome, { relaunched: [VBS], skipped: [] })
    assert.deepEqual(spawned, [[WSCRIPT, [VBS], { cwd: 'C:\\Users\\u' }]])
    assert.equal(files.has(stoppedPluginHostsPath(HOME)), false)
    assert.ok(log.some(line => line.includes('relaunched plugin service host')))

    // Idempotent: nothing left to relaunch.
    assert.deepEqual(restoreStoppedDesktopPluginHosts(HOME, deps), { relaunched: [], skipped: [] })
  })

  it('refuses to record or relaunch a launch line that is not a desktop-plugins script host', () => {
    const files = new Map<string, string>()
    const { deps, spawned } = memoryFs(new Set([VBS]), files)

    assert.equal(recordStoppedDesktopPluginHost(HOME, host({ argv: ['C:\\evil.exe', VBS] }), deps), false)
    assert.equal(files.size, 0)

    files.set(
      stoppedPluginHostsPath(HOME),
      JSON.stringify({
        schemaVersion: STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION,
        hosts: [{ ...host({ argv: ['C:\\evil.exe', VBS] }), stoppedAt: 1 }]
      })
    )

    assert.deepEqual(restoreStoppedDesktopPluginHosts(HOME, deps), { relaunched: [], skipped: [VBS] })
    assert.deepEqual(spawned, [])
    assert.equal(files.has(stoppedPluginHostsPath(HOME)), false)
  })

  it('ignores a corrupt or foreign-schema ledger and is a no-op off Windows', () => {
    const files = new Map<string, string>([[stoppedPluginHostsPath(HOME), '{not json']])
    const { deps, spawned } = memoryFs(new Set([VBS]), files)

    assert.deepEqual(restoreStoppedDesktopPluginHosts(HOME, deps), { relaunched: [], skipped: [] })
    assert.deepEqual(spawned, [])

    files.set(stoppedPluginHostsPath(HOME), JSON.stringify({ schemaVersion: 99, hosts: [{ ...host(), stoppedAt: 1 }] }))
    assert.deepEqual(restoreStoppedDesktopPluginHosts(HOME, deps), { relaunched: [], skipped: [] })

    files.set(
      stoppedPluginHostsPath(HOME),
      JSON.stringify({ schemaVersion: STOPPED_PLUGIN_HOSTS_SCHEMA_VERSION, hosts: [{ ...host(), stoppedAt: 1 }] })
    )
    assert.deepEqual(restoreStoppedDesktopPluginHosts(HOME, { ...deps, isWindows: false }), { relaunched: [], skipped: [] })
    assert.equal(files.has(stoppedPluginHostsPath(HOME)), true, 'a non-Windows call leaves the ledger alone')
  })
})
