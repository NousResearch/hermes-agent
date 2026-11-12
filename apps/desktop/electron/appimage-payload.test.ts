import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import {
  collectStalePayloadCopies,
  createStablePayloadResources,
  gatewayUnitDirs,
  type PayloadCopyHost,
  payloadCopyKey,
  payloadCopyPlan,
  preparePayloadCopy,
  runnablePayloadResources,
  tallyTree
} from './appimage-payload'
import type { InstallStamp } from './install-stamp'

const posix = test.runIf(process.platform !== 'win32')
const linux = test.runIf(process.platform === 'linux')
const roots: string[] = []

afterEach((): void => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function scratch(): string {
  const root: string = fs.mkdtempSync(path.join(os.tmpdir(), 'appimage-payload-'))
  roots.push(root)

  return root
}

function stamp(overrides: Partial<InstallStamp> = {}): InstallStamp {
  return {
    schemaVersion: 1,
    commit: '7b0a2cdbdffe3d547a35d647d52ebf5758d6c17d',
    commitDate: null,
    branch: null,
    builtAt: '2026-11-12T08:38:50.467Z',
    dirty: false,
    source: 'commit-build',
    distribution: 'desktop-app',
    updateMechanism: 'external',
    baseVersion: '0.0.0',
    displayVersion: '0.0.0',
    distance: null,
    payload: 'bundled',
    runtime: {
      repoDir: 'hermes-agent',
      toolsDir: 'tools',
      storePython: 'tools/python/bin/python3',
      sitePackages: 'venv/lib/site-packages',
      commands: { hermes: 'bin/hermes-7b0a2cd' }
    },
    tag: null,
    ...overrides
  }
}

/** A mounted artifact's resources dir: a launcher, an interpreter, and a relative venv link. */
function mountedResources(root: string): string {
  const resources: string = path.join(root, 'mount', 'resources')
  const payload: string = path.join(resources, 'agent-payload')
  fs.mkdirSync(path.join(payload, 'bin'), { recursive: true })
  fs.mkdirSync(path.join(payload, 'tools', 'python', 'bin'), { recursive: true })
  fs.mkdirSync(path.join(payload, 'venv', 'bin'), { recursive: true })
  fs.mkdirSync(path.join(payload, 'hermes-agent', 'empty'), { recursive: true })
  fs.writeFileSync(path.join(payload, 'bin', 'hermes-7b0a2cd'), '#!/bin/sh\necho launcher\n', { mode: 0o755 })
  fs.writeFileSync(path.join(payload, 'tools', 'python', 'bin', 'python3'), 'interpreter bytes', { mode: 0o755 })
  fs.writeFileSync(path.join(payload, 'hermes-agent', 'module.py'), 'print("hi")\n')
  fs.symlinkSync('../../tools/python/bin/python3', path.join(payload, 'venv', 'bin', 'python'))

  return resources
}

/** An AppImage runtime stand-in: `--appimage-extract <pattern>` unpacks into ./squashfs-root, then runs `after`. */
function fakeAppImage(root: string, resources: string, after = ''): string {
  const appimage: string = path.join(root, `Hermes-${roots.length}-${Math.random().toString(36).slice(2)}.AppImage`)

  fs.writeFileSync(
    appimage,
    [
      '#!/bin/sh',
      `[ "$1" = --appimage-extract ] && [ "$2" = 'resources/agent-payload/*' ] || exit 64`,
      `echo "$PWD" >> '${path.join(root, 'extractions')}'`,
      `mkdir -p squashfs-root/resources && cp -a '${path.join(resources, 'agent-payload')}' squashfs-root/resources/`,
      after
    ].join('\n'),
    { mode: 0o755 }
  )

  return appimage
}

function extractions(root: string): string[] {
  return fs.existsSync(path.join(root, 'extractions'))
    ? fs.readFileSync(path.join(root, 'extractions'), 'utf8').trim().split('\n')
    : []
}

function copyHost(root: string, resources: string, overrides: Partial<PayloadCopyHost> = {}): PayloadCopyHost {
  return {
    plan: {
      appimage: fakeAppImage(root, resources),
      source: path.join(resources, 'agent-payload'),
      root: path.join(root, 'data', 'hermes', 'bundles'),
      key: payloadCopyKey(stamp())!
    },
    binDir: path.join(root, 'home', '.local', 'bin'),
    log: (): void => {},
    freeBytes: async (): Promise<number> => Number.MAX_SAFE_INTEGER,
    livePaths: async (): Promise<string[]> => [],
    ...overrides
  }
}

function fuseMountinfo(mountPoint: string): string {
  return [
    '22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw',
    `640 22 0:77 / ${mountPoint.replace(/ /g, '\\040')} ro,nosuid,nodev,relatime shared:350 - fuse.Hermes.AppImage Hermes.AppImage ro,user_id=1000`
  ].join('\n')
}

test('each build gets its own copy key, and a stamp that cannot tell builds apart gets none', (): void => {
  const first: string | null = payloadCopyKey(stamp())
  const rebuilt: string | null = payloadCopyKey(stamp({ builtAt: '2026-11-12T09:01:02.003Z' }))

  assert.ok(first && rebuilt)
  assert.notEqual(first, rebuilt)
  assert.notEqual(first, payloadCopyKey(stamp({ commit: '11f2b03734c991318566575ac8dde0350c9c743c' })))
  assert.match(first, /^[\w.+-]+$/)
  assert.equal(payloadCopyKey(stamp({ builtAt: null })), null)
  assert.equal(payloadCopyKey(stamp({ commit: '0'.repeat(40) })), null)
  assert.equal(payloadCopyKey(stamp({ source: 'fallback' })), null)
})

test('cleanup looks for gateway units where `hermes gateway install` writes them', (): void => {
  assert.deepEqual(gatewayUnitDirs({}, '/home/u'), ['/home/u/.config/systemd/user', '/etc/systemd/system'])
  assert.deepEqual(gatewayUnitDirs({ XDG_CONFIG_HOME: '/cfg' }, '/home/u'), [
    '/cfg/systemd/user',
    '/etc/systemd/system'
  ])
})

posix('only a payload on the AppImage FUSE mount is copied, into the XDG data home', (): void => {
  const resources = '/tmp/.mount_Hermes Ab12/resources'

  const host = {
    env: { APPIMAGE: '/home/u/Apps/Hermes.AppImage', XDG_DATA_HOME: '/data' },
    home: '/home/u',
    resourcesPath: resources,
    stamp: stamp(),
    log: (): void => {}
  }

  assert.deepEqual(payloadCopyPlan(host, fuseMountinfo('/tmp/.mount_Hermes Ab12')), {
    appimage: '/home/u/Apps/Hermes.AppImage',
    source: path.join(resources, 'agent-payload'),
    root: '/data/hermes/bundles',
    key: payloadCopyKey(stamp())
  })
  assert.equal(
    payloadCopyPlan({ ...host, env: { XDG_DATA_HOME: 'relative' } }, fuseMountinfo('/tmp/.mount_Hermes Ab12'))?.root,
    '/home/u/.local/share/hermes/bundles'
  )
  assert.equal(payloadCopyPlan(host, fuseMountinfo('/tmp/.mount_Other')), null)

  const extracted =
    '22 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n30 22 0:30 / /tmp rw - tmpfs tmpfs rw'

  assert.equal(payloadCopyPlan({ ...host, resourcesPath: '/tmp/appimage_extracted_1/resources' }, extracted), null)
})

test('installs that are not a mounted AppImage keep running the artifact payload', async (): Promise<void> => {
  let mountinfoReads = 0

  const host = {
    platform: 'linux' as NodeJS.Platform,
    execPath: '/tmp/.mount_Hermes/hermes',
    env: { APPIMAGE: '/home/u/Hermes.AppImage' },
    home: '/home/u',
    resourcesPath: '/tmp/.mount_Hermes/resources',
    stamp: stamp(),
    readMountinfo: async (): Promise<string> => {
      mountinfoReads += 1

      return fuseMountinfo('/tmp/.mount_Hermes')
    },
    binDir: '/home/u/.local/bin',
    log: (): void => {}
  }

  for (const variant of [
    { ...host, platform: 'darwin' as NodeJS.Platform },
    { ...host, env: {} },
    // A .deb Hermes started from a terminal AppImage inherits that image's APPIMAGE and APPDIR.
    { ...host, execPath: '/opt/Hermes/hermes', env: { APPIMAGE: '/home/u/Term.AppImage', APPDIR: '/tmp/.mount_Term' } },
    { ...host, stamp: stamp({ payload: 'light' }) },
    { ...host, stamp: null }
  ]) {
    assert.equal(await runnablePayloadResources(variant), host.resourcesPath)
  }

  assert.equal(mountinfoReads, 0)
})

posix(
  'the first launch of a build unpacks a verified, relocatable copy; later launches reuse it',
  async (): Promise<void> => {
    const root: string = scratch()
    const resources: string = mountedResources(root)
    const progress: string[] = []
    const host: PayloadCopyHost = copyHost(root, resources, { progress: (phase: string) => void progress.push(phase) })

    const build: string | null = await preparePayloadCopy(host)

    assert.equal(build, path.join(host.plan.root, host.plan.key))
    const copied: string = path.join(build, 'agent-payload')
    assert.deepEqual(await tallyTree(copied), await tallyTree(host.plan.source))
    assert.equal(fs.readlinkSync(path.join(copied, 'venv', 'bin', 'python')), '../../tools/python/bin/python3')
    assert.equal(fs.statSync(path.join(copied, 'bin', 'hermes-7b0a2cd')).mode & 0o111, 0o111)
    assert.deepEqual(fs.readdirSync(build).sort(), ['agent-payload', 'complete.json'])
    assert.deepEqual(fs.readdirSync(host.plan.root), [host.plan.key])

    fs.rmSync(resources, { recursive: true })
    assert.equal(fs.readFileSync(path.join(copied, 'venv', 'bin', 'python'), 'utf8'), 'interpreter bytes')

    assert.equal(await preparePayloadCopy(host), build)
    assert.equal(extractions(root).length, 1)
    assert.deepEqual(progress, ['backend.payload'])
  }
)

posix('when the AppImage cannot unpack itself, the payload is copied from the mount', async (): Promise<void> => {
  const root: string = scratch()
  const resources: string = mountedResources(root)
  const lines: string[] = []

  const host: PayloadCopyHost = copyHost(root, resources, { log: line => void lines.push(line) })
  host.plan.appimage = fakeAppImage(root, resources, 'exit 3')

  const build: string | null = await preparePayloadCopy(host)

  assert.equal(build, path.join(host.plan.root, host.plan.key))
  assert.deepEqual(await tallyTree(path.join(build, 'agent-payload')), await tallyTree(host.plan.source))
  assert.deepEqual(fs.readdirSync(build).sort(), ['agent-payload', 'complete.json'])
  assert.match(lines.join('\n'), /could not unpack its payload .*exited 3/)
})

posix('a failed unpack is cleared before the payload is copied from the mount', async (): Promise<void> => {
  const root: string = scratch()
  const resources: string = mountedResources(root)
  const host: PayloadCopyHost = copyHost(root, resources)
  // The runtime wrote the whole payload, then failed.
  host.plan.appimage = fakeAppImage(root, resources, 'exit 3')
  const cp = fs.promises.cp
  const leftovers: string[][] = []

  const spy = vi.spyOn(fs.promises, 'cp').mockImplementation(async (from, to, options) => {
    const unpacked: string = path.join(path.dirname(String(to)), 'squashfs-root')
    leftovers.push(fs.existsSync(unpacked) ? fs.readdirSync(unpacked) : [])

    return cp(from, to, options)
  })

  try {
    assert.equal(await preparePayloadCopy(host), path.join(host.plan.root, host.plan.key))
  } finally {
    spy.mockRestore()
  }

  assert.deepEqual(leftovers, [[]])
})

posix('a short or altered copy is never used and leaves no build behind', async (): Promise<void> => {
  const root: string = scratch()
  const resources: string = mountedResources(root)

  for (const damage of [
    'rm squashfs-root/resources/agent-payload/hermes-agent/module.py',
    'printf truncated > squashfs-root/resources/agent-payload/hermes-agent/module.py',
    'rm squashfs-root/resources/agent-payload/venv/bin/python'
  ]) {
    const lines: string[] = []
    const host: PayloadCopyHost = copyHost(root, resources, { log: line => void lines.push(line) })
    host.plan.appimage = fakeAppImage(root, resources, damage)

    assert.equal(await preparePayloadCopy(host), null)
    assert.deepEqual(fs.readdirSync(host.plan.root), [])
    assert.match(lines.join('\n'), /running from the AppImage mount: Error: the copy differs/)
  }
})

posix('a build directory without a completion marker is never run', async (): Promise<void> => {
  const root: string = scratch()
  const resources: string = mountedResources(root)
  const host: PayloadCopyHost = copyHost(root, resources)
  const build: string = path.join(host.plan.root, host.plan.key)
  fs.mkdirSync(path.join(build, 'agent-payload'), { recursive: true })

  assert.equal(await preparePayloadCopy(host), null)
  assert.deepEqual(fs.readdirSync(path.join(build, 'agent-payload')), [])
  assert.deepEqual(extractions(root), [])
})

posix('concurrent first launches commit one copy and both run it', async (): Promise<void> => {
  const root: string = scratch()
  const resources: string = mountedResources(root)

  const builds: (string | null)[] = await Promise.all(
    [101, 202].map(pid => preparePayloadCopy(copyHost(root, resources, { pid })))
  )

  const host: PayloadCopyHost = copyHost(root, resources)
  assert.deepEqual(builds, [path.join(host.plan.root, host.plan.key), path.join(host.plan.root, host.plan.key)])
  assert.deepEqual(fs.readdirSync(host.plan.root), [host.plan.key])
  assert.equal(new Set(extractions(root)).size, 2)
})

posix(
  'without room for the copy, unused older copies go first; with none, the mount stays',
  async (): Promise<void> => {
    const root: string = scratch()
    const resources: string = mountedResources(root)
    const bundles: string = copyHost(root, resources).plan.root
    const older: string = path.join(bundles, 'older-build')
    fs.mkdirSync(path.join(older, 'agent-payload'), { recursive: true })
    fs.writeFileSync(path.join(older, 'complete.json'), '{"key":"older-build"}')
    const freeBytes = async (): Promise<number> => (fs.existsSync(older) ? 1024 : Number.MAX_SAFE_INTEGER)

    const build: string | null = await preparePayloadCopy(copyHost(root, resources, { freeBytes }))

    assert.ok(build)
    assert.equal(fs.existsSync(older), false)

    const fresh: string = scratch()
    const lines: string[] = []

    const tight: PayloadCopyHost = copyHost(fresh, mountedResources(fresh), {
      freeBytes: async (): Promise<number> => 1024,
      log: line => void lines.push(line)
    })

    assert.equal(await preparePayloadCopy(tight), null)
    assert.match(lines.join('\n'), /needs [\d.]+ GiB free/)
    assert.deepEqual(fs.readdirSync(tight.plan.root), [])
    assert.deepEqual(extractions(fresh), [])
  }
)

posix(
  'cleanup keeps the current and previous builds and any build a process or gateway unit runs from',
  async (): Promise<void> => {
    const root: string = scratch()
    const host: PayloadCopyHost = copyHost(root, mountedResources(root))
    const binDir: string = host.binDir
    fs.mkdirSync(binDir, { recursive: true })

    const builds: Record<string, string> = {}

    for (const [name, age] of [
      [host.plan.key, 0],
      ['previous', 10],
      ['old-running', 20],
      ['old-unit', 25],
      ['old', 30]
    ] as const) {
      builds[name] = path.join(host.plan.root, name)
      fs.mkdirSync(path.join(builds[name], 'agent-payload', 'bin'), { recursive: true })
      fs.writeFileSync(path.join(builds[name], 'complete.json'), JSON.stringify({ key: name }))
      const used = new Date(Date.now() - age * 60_000)
      fs.utimesSync(path.join(builds[name], 'complete.json'), used, used)
    }

    fs.symlinkSync(path.join(builds.old, 'agent-payload', 'bin', 'hermes-old'), path.join(binDir, 'hermes-old'))
    fs.symlinkSync('/usr/bin/env', path.join(binDir, 'unrelated'))
    // A stopped gateway service generated from an older copy: nothing runs, but its unit names the copy.
    const units: string = path.join(root, 'config', 'systemd', 'user')
    fs.mkdirSync(units, { recursive: true })
    const python: string = path.join(builds['old-unit'], 'agent-payload', 'tools', 'python', 'bin', 'python3')
    fs.writeFileSync(
      path.join(units, 'hermes-gateway-work.service'),
      `[Service]\nExecStart=${python} -m hermes_cli.main gateway run\n`
    )
    fs.writeFileSync(path.join(units, 'other.service'), `[Service]\nExecStart=${path.join(builds.old, 'x')}\n`)
    const deadWriter: string = path.join(host.plan.root, '.partial-999999999-crashed')
    const liveWriter: string = path.join(host.plan.root, `.partial-${process.pid}-copying`)
    fs.mkdirSync(deadWriter)
    fs.mkdirSync(liveWriter)

    const removed: string[] = await collectStalePayloadCopies(
      {
        ...host,
        livePaths: async () => [path.join(fs.realpathSync(builds['old-running']), 'agent-payload', 'tools', 'python3')],
        serviceUnitDirs: [units, path.join(root, 'no-such-dir')]
      },
      true
    )

    assert.deepEqual(removed.sort(), [builds.old, deadWriter].sort())
    assert.deepEqual(
      fs.readdirSync(host.plan.root).sort(),
      [host.plan.key, 'previous', 'old-running', 'old-unit', path.basename(liveWriter)].sort()
    )
    assert.deepEqual(fs.readdirSync(binDir), ['unrelated'])
  }
)

posix('cleanup reads a unit path the way systemd quoting spells it', async (): Promise<void> => {
  const root: string = scratch()
  const host: PayloadCopyHost = copyHost(root, mountedResources(root))
  // XDG_DATA_HOME=<root>/data 100%$"q\b; _systemd_command writes it as data 100%%$$\"q\\b.
  const plan = { ...host.plan, root: path.join(root, 'data 100%$"q\\b', 'hermes', 'bundles') }
  const older: string = path.join(plan.root, 'older')
  fs.mkdirSync(path.join(older, 'agent-payload'), { recursive: true })
  fs.writeFileSync(path.join(older, 'complete.json'), JSON.stringify({ key: 'older' }))
  const units: string = path.join(root, 'units')
  fs.mkdirSync(units)

  const python: string = path.join(
    root,
    'data 100%%$$\\"q\\\\b',
    'hermes',
    'bundles',
    'older',
    'agent-payload',
    'python3'
  )

  fs.writeFileSync(
    path.join(units, 'hermes-gateway.service'),
    `[Service]\nExecStart="${python}" -m hermes_cli.main gateway run\n`
  )

  assert.deepEqual(await collectStalePayloadCopies({ ...host, plan, serviceUnitDirs: [units] }, false), [])
  assert.ok(fs.existsSync(older))
})

linux('cleanup keeps every build an open Desktop selected, even when short of space', async (): Promise<void> => {
  const root: string = scratch()
  // No livePaths stub: cleanup reads this process's /proc entries, as another launch would.
  const { livePaths: _stub, ...host }: PayloadCopyHost = copyHost(root, mountedResources(root))
  const build = (key: string): string => path.join(host.plan.root, key)

  for (const key of ['reused', 'unused']) {
    fs.mkdirSync(path.join(build(key), 'agent-payload'), { recursive: true })
    fs.writeFileSync(path.join(build(key), 'complete.json'), JSON.stringify({ key }))
  }

  // Two open Desktops of other builds, neither running Python yet: one reused its copy, one made it.
  assert.equal(await preparePayloadCopy({ ...host, plan: { ...host.plan, key: 'reused' } }), build('reused'))
  assert.equal(await preparePayloadCopy(host), build(host.plan.key))

  const removed: string[] = await collectStalePayloadCopies({ ...host, plan: { ...host.plan, key: 'cleaner' } }, false)

  assert.deepEqual(removed, [build('unused')])
  assert.deepEqual(fs.readdirSync(host.plan.root).sort(), [host.plan.key, 'reused'].sort())
})

posix('a copy removed while the app runs is prepared again, not handed out', async (): Promise<void> => {
  const root: string = scratch()
  const resources: string = mountedResources(root)
  const appimage: string = fakeAppImage(root, resources)
  const lines: string[] = []
  const log = (line: string): void => void lines.push(line)

  const stable = createStablePayloadResources(hostLog => ({
    platform: 'linux',
    execPath: path.join(root, 'mount', 'hermes'),
    env: { APPIMAGE: appimage, APPDIR: path.join(root, 'mount'), XDG_DATA_HOME: path.join(root, 'data') },
    home: path.join(root, 'home'),
    resourcesPath: resources,
    stamp: stamp(),
    readMountinfo: async (): Promise<string> => fuseMountinfo(path.join(root, 'mount')),
    binDir: path.join(root, 'home', '.local', 'bin'),
    log: hostLog,
    freeBytes: async (): Promise<number> => Number.MAX_SAFE_INTEGER,
    livePaths: async (): Promise<string[]> => []
  }))

  const build: string = await stable(log)
  assert.equal(build, path.join(root, 'data', 'hermes', 'bundles', payloadCopyKey(stamp())!))
  assert.equal(await stable(log), build)
  assert.equal(extractions(root).length, 1)

  fs.rmSync(build, { recursive: true })

  assert.equal(await stable(log), build)
  assert.deepEqual(
    await tallyTree(path.join(build, 'agent-payload')),
    await tallyTree(path.join(resources, 'agent-payload'))
  )
  assert.equal(extractions(root).length, 2)
  assert.match(lines.join('\n'), /is gone; preparing this build's copy again/)
})
