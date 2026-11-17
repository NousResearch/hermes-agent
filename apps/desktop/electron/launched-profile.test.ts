import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, expect, it, vi } from 'vitest'

import {
  normalizeLauncherUrl,
  normalizeProfileLauncher,
  profileHasRemoteConnection,
  profileIsLaunched,
  profileLauncher,
  profileRemoteOverride,
  resolveProfileBackendRoute
} from './connection-config'
import { sanitizeConnectionProfiles } from './connection-profiles'
import { migrateV1ToRegistry } from './connection-registry'
import { migrateProfileScopedDesktopPlugins, reconcileUnifiedDesktopHalves } from './desktop-plugins-root'
import { resolveDesktopRemoteRoute } from './desktop-remote-route'
import { createLaunchedProfileHost, launchedProfileHostRequest } from './launched-profile-host'
import { assertProfileMayRunLocally, PROFILES_RUN_ELSEWHERE_DIR, readRunElsewhereMarker } from './profile-runs-elsewhere'

// A profile a desktop plugin's launcher runs (host.setProfileBackend) is a
// per-profile `mode: 'remote'` entry carrying `launcher`.

const TOKEN = { encoding: 'plain', value: 'launcher-token' }

const launchedConfig = (extra: Record<string, unknown> = {}) => ({
  mode: 'local',
  remote: {},
  profiles: {
    work: {
      mode: 'remote',
      url: 'http://127.0.0.1:47911',
      authMode: 'token',
      token: TOKEN,
      launcher: { plugin: 'openshell' }
    },
    ...extra
  }
})

const dirs: string[] = []

function tempDir(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix))
  dirs.push(dir)

  return dir
}

function write(file: string, text: string) {
  fs.mkdirSync(path.dirname(file), { recursive: true })
  fs.writeFileSync(file, text)
}

afterEach(() => {
  for (const dir of dirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

describe('launcher marker', () => {
  it('rides on the remote override and identifies a launched profile', () => {
    const config = launchedConfig({ research: { mode: 'remote', url: 'https://box.example', token: TOKEN } })

    expect(profileRemoteOverride(config, 'work')?.launcher).toEqual({ plugin: 'openshell' })
    expect(profileLauncher(config, 'work')).toEqual({ plugin: 'openshell' })
    expect(profileIsLaunched(config, 'work')).toBe(true)
    expect(profileHasRemoteConnection(config, 'work')).toBe(true)
    expect(profileIsLaunched(config, 'research')).toBe(false)
    expect(profileRemoteOverride(config, 'research')).not.toHaveProperty('launcher')
    expect(profileIsLaunched(config, 'default')).toBe(false)
  })

  it('ignores a marker on a cloud entry or with a malformed plugin id', () => {
    const config = {
      profiles: {
        cloudy: { mode: 'cloud', url: 'https://x.example', launcher: { plugin: 'openshell' } },
        odd: { mode: 'remote', url: 'http://127.0.0.1:1', launcher: { plugin: '../evil' } }
      }
    }

    expect(profileIsLaunched(config, 'cloudy')).toBe(false)
    expect(profileIsLaunched(config, 'odd')).toBe(false)
    expect(normalizeProfileLauncher({ plugin: 'Openshell' })).toBeNull()
    expect(normalizeProfileLauncher('openshell')).toBeNull()
    expect(normalizeProfileLauncher({ plugin: ' openshell ' })).toEqual({ plugin: 'openshell' })
  })

  it('survives the read of connection.json on a remote entry only', () => {
    const profiles: Record<string, unknown> = sanitizeConnectionProfiles({
      ...launchedConfig().profiles,
      local: { mode: 'local', launcher: { plugin: 'openshell' } },
      odd: { mode: 'sandbox', url: 'http://127.0.0.1:1', launcher: { plugin: 'openshell' } }
    })

    expect(profiles.work).toMatchObject({ mode: 'remote', launcher: { plugin: 'openshell' } })
    expect(profiles.local).not.toHaveProperty('launcher')
    // An unknown mode reads as local: never a launched profile, never a silent local start of one.
    expect(profiles.odd).toMatchObject({ mode: 'local' })
    expect(profiles.odd).not.toHaveProperty('launcher')
  })

  it('accepts only loopback launcher URLs', () => {
    expect(normalizeLauncherUrl('http://127.0.0.1:47911/')).toBe('http://127.0.0.1:47911')
    expect(normalizeLauncherUrl('127.0.0.1:47911')).toBe('http://127.0.0.1:47911')
    expect(normalizeLauncherUrl('http://localhost:47911')).toBe('http://localhost:47911')
    expect(normalizeLauncherUrl('http://[::1]:47911')).toBe('http://[::1]:47911')
    expect(() => normalizeLauncherUrl('http://192.168.1.4:47911')).toThrow(/this computer/)
    expect(() => normalizeLauncherUrl('https://box.example')).toThrow(/this computer/)
    expect(() => normalizeLauncherUrl('http://127.0.0.1.nip.io:47911')).toThrow(/this computer/)
    expect(() => normalizeLauncherUrl('file:///etc/passwd')).toThrow()
  })

  it('names the listing calls the host answers for a launched profile', () => {
    expect(launchedProfileHostRequest('GET', '/api/profiles')).toBe('list')
    expect(launchedProfileHostRequest(undefined, '/api/profiles?include_sessions=0')).toBe('list')
    expect(launchedProfileHostRequest('GET', '/api/profiles/active')).toBe('active')
    expect(launchedProfileHostRequest('POST', '/api/profiles')).toBeNull()
    expect(launchedProfileHostRequest('GET', '/api/profiles/sessions')).toBeNull()
    expect(launchedProfileHostRequest('GET', '/api/sessions')).toBeNull()
  })
})

describe('routing under a launched primary', () => {
  it('keeps the launched primary on its launcher and gives every other local profile this host', () => {
    // A launched primary resolves remote, which on its own would route other profiles through it.
    const opts = { primaryLaunched: true, primaryRemoteActive: true, primaryProfile: 'work', requestMethod: 'GET' }

    expect(resolveProfileBackendRoute('work', { ...opts, requestPath: '/api/config' })).toMatchObject({
      backend: 'primary'
    })

    for (const requestPath of ['/api/config', '/api/profiles', '/api/sessions', undefined]) {
      expect(resolveProfileBackendRoute('notes', { ...opts, requestPath })).toEqual({
        backend: 'pool',
        descriptorProfile: null,
        scopePath: false
      })
    }

    // Without a launched primary the host backend is shared by scope.
    expect(
      resolveProfileBackendRoute('notes', {
        primaryProfile: 'default',
        requestMethod: 'GET',
        requestPath: '/api/config'
      })
    ).toEqual({ backend: 'primary', descriptorProfile: 'notes', scopePath: true })
  })
})

describe('registry', () => {
  it('does not migrate a launcher entry into a registered gateway', () => {
    const registry = migrateV1ToRegistry(
      launchedConfig({ research: { mode: 'remote', url: 'https://box.example:9119', token: TOKEN } })
    )

    expect(registry.connections.map(connection => connection.label)).toEqual(['This device', 'box.example:9119'])
    expect(registry.primary).toBe('local')
  })

  it('routes a launched profile to its launcher without a registry identity', () => {
    const v1 = launchedConfig()

    const registry = {
      version: 2 as const,
      launchMode: 'primary' as const,
      primary: 'local',
      lastUsed: 'local',
      connections: [
        { id: 'local', kind: 'local' as const, label: 'This device' },
        { id: 'ghost', kind: 'remote' as const, label: 'Sandbox', url: 'http://127.0.0.1:47911', token: TOKEN }
      ]
    }

    const route = resolveDesktopRemoteRoute({ config: v1, profile: 'work', registry })

    expect(route).toMatchObject({ kind: 'remote', launcher: { plugin: 'openshell' }, url: 'http://127.0.0.1:47911' })
    expect(route).not.toHaveProperty('connectionId')
  })
})

describe('desktop plugin root', () => {
  it('never copies a desktop half or a scoped plugin out of a launched home', async () => {
    const home = tempDir('hermes-launched-root-')
    const appRoot = path.join(home, 'desktop-plugins')

    write(path.join(home, 'plugins', 'safe', 'desktop', 'plugin.js'), 'export default {}')
    write(path.join(home, 'profiles', 'work', 'plugins', 'planted', 'desktop', 'plugin.js'), 'export default {}')
    write(path.join(home, 'profiles', 'work', 'desktop-plugins', 'scoped', 'plugin.js'), 'export default {}')
    write(path.join(home, 'profiles', 'other', 'plugins', 'fine', 'desktop', 'plugin.js'), 'export default {}')
    write(path.join(home, 'profiles', 'other', 'desktop-plugins', 'moved', 'plugin.js'), 'export default {}')
    // The scan's caller creates the app root first (fs-ipc.ts desktopPluginsRoot); without it nothing can move.
    fs.mkdirSync(appRoot)

    const isLaunched = (profile: string) => profile === 'work'

    await migrateProfileScopedDesktopPlugins(home, appRoot, isLaunched)
    const touched = await reconcileUnifiedDesktopHalves(home, appRoot, isLaunched)

    expect(fs.existsSync(path.join(appRoot, 'moved', 'plugin.js'))).toBe(true)
    expect(touched.map(dir => path.basename(dir)).sort()).toEqual(['fine', 'safe'])
    expect(fs.existsSync(path.join(appRoot, 'planted'))).toBe(false)
    expect(fs.existsSync(path.join(appRoot, 'scoped'))).toBe(false)
    expect(fs.existsSync(path.join(home, 'profiles', 'work', 'desktop-plugins', 'scoped', 'plugin.js'))).toBe(true)
  })

  it('stops refreshing a copy whose source home became launched', async () => {
    const home = tempDir('hermes-launched-refresh-')
    const appRoot = path.join(home, 'desktop-plugins')
    const source = path.join(home, 'profiles', 'work', 'plugins', 'tool', 'desktop', 'plugin.js')

    write(source, 'export default { v: 1 }')
    await reconcileUnifiedDesktopHalves(home, appRoot)

    write(source, 'export default { v: 2 }')
    fs.utimesSync(source, new Date(), new Date(Date.now() + 60_000))

    expect(await reconcileUnifiedDesktopHalves(home, appRoot, profile => profile === 'work')).toEqual([])
    expect(fs.readFileSync(path.join(appRoot, 'tool', 'plugin.js'), 'utf8')).toBe('export default { v: 1 }')
  })
})

describe('profiles_run_elsewhere marker', () => {
  it('marks nothing without the folder and reads the plugin from a marker', () => {
    const root = tempDir('hermes-run-elsewhere-')

    expect(readRunElsewhereMarker(root, 'work')).toBeNull()

    write(
      path.join(root, PROFILES_RUN_ELSEWHERE_DIR, 'work.json'),
      '{"plugin": "openshell", "since": "2026-10-26T18:03:11Z"}'
    )

    expect(readRunElsewhereMarker(root, 'work')).toEqual({ plugin: 'openshell' })
    expect(readRunElsewhereMarker(root, 'Work')).toEqual({ plugin: 'openshell' })
    expect(readRunElsewhereMarker(root, 'other')).toBeNull()
    expect(readRunElsewhereMarker(root, null)).toBeNull()
  })

  it('reads an unreadable marker as marked, as core does', () => {
    const root = tempDir('hermes-run-elsewhere-bad-')

    write(path.join(root, PROFILES_RUN_ELSEWHERE_DIR, 'default.json'), 'not json')
    expect(readRunElsewhereMarker(root, 'default')).toEqual({ plugin: null })
    expect(readRunElsewhereMarker(root, '')).toEqual({ plugin: null })
  })

  it('marks every profile when a file sits where the marker folder belongs, as core does', () => {
    const root = tempDir('hermes-run-elsewhere-file-')

    // A missing folder marks nothing. A file in its place cannot be listed, so the app cannot tell
    // which profiles it may start (core: NotADirectoryError refuses every profile).
    write(path.join(root, PROFILES_RUN_ELSEWHERE_DIR), 'a file, not a folder')

    expect(readRunElsewhereMarker(root, 'work')).toEqual({ plugin: null })
    expect(readRunElsewhereMarker(root, 'default')).toEqual({ plugin: null })
    expect(() => assertProfileMayRunLocally(root, 'work')).toThrow(/"work" runs through a plugin,/)
  })

  it('refuses a local start of a marked profile, naming the profile and the plugin', () => {
    const root = tempDir('hermes-run-elsewhere-refuse-')

    expect(() => assertProfileMayRunLocally(root, 'work')).not.toThrow()

    write(path.join(root, PROFILES_RUN_ELSEWHERE_DIR, 'work.json'), '{"plugin": "openshell"}')
    write(path.join(root, PROFILES_RUN_ELSEWHERE_DIR, 'notes.json'), '')

    expect(() => assertProfileMayRunLocally(root, 'work')).toThrow(
      'Profile "work" runs through the openshell plugin, but this app has no connection to it. Hermes won\'t start it on this computer.'
    )
    expect(() => assertProfileMayRunLocally(root, 'notes')).toThrow(/"notes" runs through a plugin,/)
  })
})

describe('the host for a launched profile', () => {
  const hostFor = (config: Record<string, any>, primary: string) => {
    const home = tempDir('hermes-launched-host-')
    const startPrimary = vi.fn(async () => 'primary-backend')
    const ensureProfileBackend = vi.fn(async (profile: string) => `${profile}-backend`)
    const fetchJson = vi.fn(async (backend: string, requestPath: string) => ({ backend, requestPath }))

    fs.mkdirSync(path.join(home, 'profiles', 'notes'), { recursive: true })
    fs.mkdirSync(path.join(home, 'profiles', 'work'), { recursive: true })

    const host = createLaunchedProfileHost({
      ensureProfileBackend,
      fetchJson,
      hermesHome: home,
      isProfileName: name => /^[a-z0-9][a-z0-9_-]{0,63}$/.test(name),
      primaryProfile: () => primary,
      readConfig: () => config,
      startPrimary
    })

    return { ensureProfileBackend, fetchJson, home, host, startPrimary }
  }

  it("answers a launched profile's listing from this computer's backend", async () => {
    const { fetchJson, host } = hostFor(launchedConfig(), 'default')

    expect(await host.answer({ path: '/api/profiles', profile: 'work' })).toEqual({
      backend: 'primary-backend',
      requestPath: '/api/profiles'
    })
    expect(await host.answer({ path: '/api/profiles/active', profile: 'work' })).toEqual({
      active: 'default',
      current: 'work'
    })
    // Not launched, or not a listing call: the normal route answers.
    expect(await host.answer({ path: '/api/profiles', profile: 'notes' })).toBeUndefined()
    expect(await host.answer({ path: '/api/sessions', profile: 'work' })).toBeUndefined()
    expect(fetchJson).toHaveBeenCalledTimes(1)
  })

  it('lists from the first profile no plugin runs when a plugin runs the primary', async () => {
    const { ensureProfileBackend, home, host, startPrimary } = hostFor(
      { profiles: { ...launchedConfig().profiles, default: launchedConfig().profiles.work } },
      'default'
    )

    write(path.join(home, PROFILES_RUN_ELSEWHERE_DIR, 'notes.json'), '{"plugin": "other"}')

    await expect(host.ensureHostBackend()).rejects.toThrow(/Every profile on this computer runs through a plugin/)

    fs.mkdirSync(path.join(home, 'profiles', 'zed'))

    expect(await host.ensureHostBackend()).toBe('zed-backend')
    expect(await host.rosterBackend('local', async () => 'registry-backend')).toBe('zed-backend')
    expect(await host.rosterBackend('remote', async () => 'registry-backend')).toBe('registry-backend')
    expect(ensureProfileBackend).toHaveBeenCalledWith('zed')
    expect(startPrimary).not.toHaveBeenCalled()
  })

  it('lists from a profile that runs on this computer, never from one connected to a remote host', async () => {
    const { ensureProfileBackend, home, host } = hostFor(
      {
        profiles: {
          default: launchedConfig().profiles.work,
          notes: { mode: 'remote', url: 'https://box.example', token: TOKEN },
          ops: { mode: 'ssh', host: 'box.example' }
        }
      },
      'default'
    )

    fs.mkdirSync(path.join(home, 'profiles', 'ops'))

    // "notes" and "ops" come first by name, but their backends are on another computer.
    expect(await host.ensureHostBackend()).toBe('work-backend')
    expect(ensureProfileBackend.mock.calls).toEqual([['work']])

    fs.rmSync(path.join(home, 'profiles', 'work'), { recursive: true })
    await expect(host.ensureHostBackend()).rejects.toThrow(/runs through a plugin or connects to a remote host/)
  })

  it('answers a listing pinned to "This device" too, and leaves a pinned remote connection alone', async () => {
    const { fetchJson, host, startPrimary } = hostFor(launchedConfig(), 'default')
    const dial = vi.fn(async () => 'this-device-backend')
    const pinned = (kind: string) => ({ dial, kind: () => kind })

    // This device's own backend answers, as the registry resolves it (the v1 primary can be remote).
    expect(await host.answer({ path: '/api/profiles', profile: 'work' }, pinned('local'))).toEqual({
      backend: 'this-device-backend',
      requestPath: '/api/profiles'
    })
    expect(await host.answer({ path: '/api/profiles/active', profile: 'work' }, pinned('local'))).toEqual({
      active: 'default',
      current: 'work'
    })
    expect(startPrimary).not.toHaveBeenCalled()

    // A registered gateway on another computer lists its own profiles, a same-named one included.
    for (const kind of ['remote', 'cloud', 'ssh']) {
      expect(await host.answer({ path: '/api/profiles', profile: 'work' }, pinned(kind))).toBeUndefined()
    }

    expect(fetchJson).toHaveBeenCalledTimes(1)
  })

  it('answers a pinned listing for a launched primary from another local profile, not through its route', async () => {
    const { host } = hostFor({ profiles: { default: launchedConfig().profiles.work } }, 'default')
    const dial = vi.fn(async () => 'the-sandbox')

    expect(await host.answer({ path: '/api/profiles' }, { dial, kind: () => 'local' })).toEqual({
      backend: 'notes-backend',
      requestPath: '/api/profiles'
    })
    expect(dial).not.toHaveBeenCalled()
  })

  it("refuses a marked profile the host backend would serve by scope, unless a remote serves it", () => {
    const { home, host } = hostFor({ profiles: {} }, 'default')

    write(path.join(home, PROFILES_RUN_ELSEWHERE_DIR, 'notes.json'), '{"plugin": "openshell"}')

    expect(() => host.assertScopeMayRunLocally('notes', {})).toThrow(/"notes" runs through the openshell plugin/)
    expect(() => host.assertScopeMayRunLocally('notes', { globalRemote: true })).not.toThrow()
    expect(() => host.assertScopeMayRunLocally('default', {})).not.toThrow()
    expect(host.homeIsSandboxed('notes')).toBe(true)
    expect(host.homeIsSandboxed('work')).toBe(false)
  })
})
