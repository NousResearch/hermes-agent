/**
 * An AppImage mounts itself at a new /tmp/.mount_* on every launch and unmounts on quit, so CLI
 * links, the gateway and service units must not run the agent payload from there. The first launch
 * of each build unpacks the payload to <XDG data>/hermes/bundles/<build>/agent-payload; every launch
 * of that build runs the copy. Other installs, and AppImages run extracted, keep their own payload.
 */
import { spawn } from 'node:child_process'
import { randomUUID } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { isFallbackCommit } from './bundle-skew'
import { removeBundleCliLinks } from './cli-provision'
import { INSTALL_STAMP, type InstallStamp } from './install-stamp'
import { ownAppImage } from './relaunch-target'

export type BootProgress = (phase: string, message: string, progress: number) => Promise<void> | void

export interface PayloadTally {
  entries: number
  files: number
  links: number
  bytes: number
}

export interface PayloadCopyPlan {
  /** The AppImage file, whose runtime unpacks the payload. */
  appimage: string
  /** The mounted artifact's agent-payload. */
  source: string
  /** <XDG data>/hermes/bundles, one directory per build. */
  root: string
  key: string
}

export interface PayloadCopyHost {
  plan: PayloadCopyPlan
  binDir: string
  log: (line: string) => void
  progress?: BootProgress
  freeBytes?: (dir: string) => Promise<number>
  /** Paths live processes run from or hold open (/proc/<pid>/exe and /proc/<pid>/fd). */
  livePaths?: () => Promise<string[]>
  /** Where `hermes gateway install` writes systemd units (see gatewayUnitDirs). */
  serviceUnitDirs?: string[]
  isAlive?: (pid: number) => boolean
  pid?: number
}

export interface RunnableResourcesHost extends Omit<PayloadCopyHost, 'plan'> {
  platform: NodeJS.Platform
  execPath: string
  env: NodeJS.ProcessEnv
  home: string
  resourcesPath: string
  stamp: Readonly<InstallStamp> | null
  readMountinfo: () => Promise<string>
}

const PAYLOAD_DIR = 'agent-payload'
const MARKER = 'complete.json'
const PARTIAL_PREFIX = '.partial-'
const BLOCK_SLACK_BYTES = 4096
const RESERVE_BYTES = 1024 ** 3

/** hermes_cli/gateway.py user_systemd_unit_dir() and _SYSTEM_UNIT_DIR. */
export function gatewayUnitDirs(env: NodeJS.ProcessEnv, home: string): string[] {
  const xdg: string | undefined = env.XDG_CONFIG_HOME?.trim()

  return [path.join(xdg || path.join(home, '.config'), 'systemd', 'user'), '/etc/systemd/system']
}

export function payloadCopiesRoot(env: NodeJS.ProcessEnv, home: string): string {
  const xdg: string | undefined = env.XDG_DATA_HOME

  return path.join(xdg && path.isAbsolute(xdg) ? xdg : path.join(home, '.local', 'share'), 'hermes', 'bundles')
}

/** One directory per build, so a rebuild at the same commit never runs an older copy. */
export function payloadCopyKey(stamp: Readonly<InstallStamp> | null): string | null {
  if (!stamp?.commit || !stamp.builtAt || stamp.source === 'fallback' || isFallbackCommit(stamp.commit)) {
    return null
  }

  const version: string = stamp.displayVersion ? `${stamp.displayVersion}-` : ''

  return `${version}${stamp.commit.slice(0, 12)}-${stamp.builtAt}`.replace(/[^\w.+-]/g, '')
}

function unescapeMountPath(value: string): string {
  return value.replace(/\\([0-7]{3})/g, (_match, octal: string) => String.fromCharCode(parseInt(octal, 8)))
}

/** The filesystem type of the innermost mount holding `target`, from /proc/self/mountinfo. */
export function mountTypeFor(target: string, mountinfo: string): string | null {
  let best: { point: string; type: string } | null = null

  for (const line of mountinfo.split('\n')) {
    const [mount, filesystem] = line.split(' - ')
    const point: string = unescapeMountPath(mount?.split(' ')[4] ?? '')
    const relative: string = path.relative(point, target)

    if (!filesystem || !point || relative.startsWith('..') || path.isAbsolute(relative)) {
      continue
    }

    if (!best || point.length >= best.point.length) {
      best = { point, type: filesystem.split(' ')[0] }
    }
  }

  return best?.type ?? null
}

/** Only a payload served from the AppImage's FUSE mount is copied; an extracted AppImage runs in place. */
export function payloadCopyPlan(
  host: Pick<RunnableResourcesHost, 'env' | 'home' | 'resourcesPath' | 'stamp' | 'log'>,
  mountinfo: string
): PayloadCopyPlan | null {
  if (!/^fuse(\.|blk$|$)/.test(mountTypeFor(host.resourcesPath, mountinfo) ?? '')) {
    return null
  }

  const key: string | null = payloadCopyKey(host.stamp)

  if (!key) {
    host.log('[payload] the install stamp cannot tell builds apart; running from the AppImage mount')

    return null
  }

  return {
    appimage: host.env.APPIMAGE!,
    source: path.join(host.resourcesPath, PAYLOAD_DIR),
    root: payloadCopiesRoot(host.env, host.home),
    key
  }
}

export async function tallyTree(root: string): Promise<PayloadTally> {
  const tally: PayloadTally = { entries: 0, files: 0, links: 0, bytes: 0 }

  const walk = async (dir: string): Promise<void> => {
    const entries: fs.Dirent[] = await fs.promises.readdir(dir, { withFileTypes: true })

    await Promise.all(
      entries.map(async (entry): Promise<void> => {
        const full: string = path.join(dir, entry.name)
        tally.entries += 1

        if (entry.isDirectory()) {
          return walk(full)
        }

        if (entry.isSymbolicLink()) {
          tally.links += 1

          return
        }

        const { size } = await fs.promises.lstat(full)
        tally.files += 1
        tally.bytes += size
      })
    )
  }

  await walk(root)

  return tally
}

const copyTree = (from: string, to: string): Promise<void> =>
  fs.promises.cp(from, to, {
    recursive: true,
    verbatimSymlinks: true,
    preserveTimestamps: true,
    errorOnExist: true,
    force: false
  })

function appImageExtract(appimage: string, cwd: string): Promise<void> {
  return new Promise((resolve, reject) => {
    const child = spawn(appimage, ['--appimage-extract', `resources/${PAYLOAD_DIR}/*`], {
      cwd,
      stdio: ['ignore', 'ignore', 'pipe']
    })

    let stderr = ''
    child.stderr.on('data', (chunk: Buffer) => void (stderr = `${stderr}${chunk}`.slice(-2000)))
    child.on('error', reject)
    child.on('close', (code: number | null) =>
      code === 0 ? resolve() : reject(new Error(`--appimage-extract exited ${code}: ${stderr.trim()}`))
    )
  })
}

/** The AppImage runtime unpacks its own squashfs over ten times faster than a copy read back through the mount. */
async function unpackPayload(host: PayloadCopyHost, target: string): Promise<void> {
  const { appimage, source } = host.plan
  const unpacked: string = path.join(path.dirname(target), 'squashfs-root')

  try {
    await appImageExtract(appimage, path.dirname(target))
    await fs.promises.rename(path.join(unpacked, 'resources', PAYLOAD_DIR), target)
  } catch (error) {
    host.log(`[payload] ${appimage} could not unpack its payload (${String(error)}); copying it from the mount`)
    // A failed unpack can hold most of the payload; the space check allowed for one tree, not two.
    await fs.promises.rm(target, { recursive: true, force: true })
    await fs.promises.rm(unpacked, { recursive: true, force: true })
    await copyTree(source, target)
  } finally {
    await fs.promises.rm(unpacked, { recursive: true, force: true })
  }
}

async function statfsFree(dir: string): Promise<number> {
  const stats: fs.StatsFs = await fs.promises.statfs(dir)

  return stats.bavail * stats.bsize
}

async function procHeldPaths(): Promise<string[]> {
  const pids: string[] = (await fs.promises.readdir('/proc')).filter(name => /^\d+$/.test(name))

  const held: string[][] = await Promise.all(
    pids.map(async (pid): Promise<string[]> => {
      const fds: string[] = await fs.promises.readdir(`/proc/${pid}/fd`).catch((): string[] => [])
      const links: string[] = [`/proc/${pid}/exe`, ...fds.map(fd => `/proc/${pid}/fd/${fd}`)]

      return Promise.all(links.map(link => fs.promises.readlink(link).catch(() => '')))
    })
  )

  return held.flat().filter(Boolean)
}

function pidAlive(pid: number): boolean {
  try {
    process.kill(pid, 0)

    return true
  } catch (error) {
    return (error as NodeJS.ErrnoException).code === 'EPERM'
  }
}

/** Installed gateway units. A stopped one runs nothing, so /proc cannot show that its ExecStart names a copy. */
async function gatewayUnits(dirs: string[]): Promise<string[]> {
  const units: string[][] = await Promise.all(
    dirs.map(async (dir): Promise<string[]> => {
      const names: string[] = await fs.promises.readdir(dir).catch((): string[] => [])

      return Promise.all(
        names
          .filter(name => name.startsWith('hermes-gateway') && name.endsWith('.service'))
          .map(name => fs.promises.readFile(path.join(dir, name), 'utf8').catch(() => ''))
      )
    })
  )

  return units.flat()
}

/** A path as hermes_cli/gateway.py _systemd_command quotes it into a unit's argv. */
function systemdQuoted(value: string): string {
  return value
    .replace(/\\/g, '\\\\')
    .replace(/"/g, '\\"')
    .replace(/%/g, '%%')
    .replace(/\$/g, () => '$$')
}

async function markedComplete(build: string, key: string): Promise<boolean> {
  try {
    return JSON.parse(await fs.promises.readFile(path.join(build, MARKER), 'utf8'))?.key === key
  } catch {
    return false
  }
}

async function lastUsed(build: string): Promise<number> {
  return (await fs.promises.stat(path.join(build, MARKER)).catch(() => null))?.mtimeMs ?? 0
}

const openFd = promisify(fs.open)
const heldCopies = new Map<string, number>()

/**
 * Hold this build's marker open for the life of the process, then check it is still complete. An
 * open Desktop runs Electron from its mount, so until its gateway starts nothing under the copy shows
 * in /proc/<pid>/exe; the open marker is what tells another launch's cleanup the copy is in use.
 */
async function holdPayloadCopy(build: string, key: string): Promise<boolean> {
  const fd: number | null = await openFd(path.join(build, MARKER), 'r').catch(() => null)

  if (fd === null || !(await markedComplete(build, key))) {
    if (fd !== null) {
      fs.close(fd, () => undefined)
    }

    return false
  }

  const previous: number | undefined = heldCopies.get(build)
  heldCopies.set(build, fd)

  if (previous !== undefined) {
    fs.close(previous, () => undefined)
  }

  return true
}

const gib = (bytes: number): string => (bytes / 1024 ** 3).toFixed(1)

/**
 * Remove copies of other builds that no live process runs from or holds open (an open Desktop holds
 * its own; see holdPayloadCopy) and no installed gateway unit runs, and partial copies whose writer
 * died. `keepPrevious` keeps the most recently used other build so switching between two
 * installed AppImages does not copy the payload again on every switch.
 */
export async function collectStalePayloadCopies(
  host: Omit<PayloadCopyHost, 'progress' | 'freeBytes' | 'pid'>,
  keepPrevious: boolean
): Promise<string[]> {
  const { root, key } = host.plan
  const alive = host.isAlive ?? pidAlive
  const held: string[] = await (host.livePaths ?? procHeldPaths)()
  const units: string[] = await gatewayUnits(host.serviceUnitDirs ?? [])
  const doomed: string[] = []
  const builds: { dir: string; used: number }[] = []

  for (const entry of await fs.promises.readdir(root, { withFileTypes: true })) {
    const dir: string = path.join(root, entry.name)

    if (!entry.isDirectory() || entry.name === key) {
      continue
    }

    if (entry.name.startsWith(PARTIAL_PREFIX)) {
      if (!alive(Number.parseInt(entry.name.slice(PARTIAL_PREFIX.length), 10))) {
        doomed.push(dir)
      }

      continue
    }

    builds.push({ dir, used: await lastUsed(dir) })
  }

  builds.sort((a, b) => b.used - a.used)

  for (const { dir } of builds.slice(keepPrevious ? 1 : 0)) {
    const real: string = await fs.promises.realpath(dir)

    if (held.some(used => used.startsWith(`${real}${path.sep}`))) {
      host.log(`[payload] keeping ${dir}: a running process uses it`)

      continue
    }

    const spellings: string[] = [dir, real].flatMap(at => [`${at}${path.sep}`, systemdQuoted(`${at}${path.sep}`)])

    if (units.some(unit => spellings.some(spelling => unit.includes(spelling)))) {
      host.log(`[payload] keeping ${dir}: a gateway service unit runs from it`)

      continue
    }

    removeBundleCliLinks(path.join(dir, PAYLOAD_DIR), host.binDir)
    doomed.push(dir)
  }

  for (const dir of doomed) {
    await fs.promises.rm(dir, { recursive: true, force: true })
    host.log(`[payload] removed stale payload copy ${dir}`)
  }

  return doomed
}

async function copyBuild(host: PayloadCopyHost, build: string, partial: string): Promise<void> {
  const { root, key, source } = host.plan
  const freeBytes = host.freeBytes ?? statfsFree
  const expected: PayloadTally = await tallyTree(source)
  const needed: number = expected.bytes + expected.entries * BLOCK_SLACK_BYTES + RESERVE_BYTES
  let free: number = await freeBytes(root)

  if (free < needed) {
    await collectStalePayloadCopies(host, false)
    free = await freeBytes(root)
  }

  if (free < needed) {
    throw new Error(`the copy needs ${gib(needed)} GiB free in ${root}, which has ${gib(free)} GiB`)
  }

  await host.progress?.('backend.payload', 'Preparing this Hermes version (first launch only)…', 28)
  host.log(`[payload] unpacking the AppImage payload (${gib(expected.bytes)} GiB) to ${build}`)
  const started: number = Date.now()
  const target: string = path.join(partial, PAYLOAD_DIR)
  await unpackPayload(host, target)
  const copied: PayloadTally = await tallyTree(target)

  if (JSON.stringify(copied) !== JSON.stringify(expected)) {
    throw new Error(`the copy differs from the payload: ${JSON.stringify({ expected, copied })}`)
  }

  await fs.promises.writeFile(path.join(partial, MARKER), `${JSON.stringify({ key, ...expected }, null, 2)}\n`)

  try {
    await fs.promises.rename(partial, build)
  } catch (error) {
    if (!(await markedComplete(build, key))) {
      throw error
    }

    // Another launch committed the same build first; its copy is as good as ours.
    await fs.promises.rm(partial, { recursive: true, force: true })
  }

  host.log(
    `[payload] copied and verified ${expected.entries} entries in ${((Date.now() - started) / 1000).toFixed(1)}s`
  )
}

/**
 * The directory holding this build's verified agent-payload copy, or null when the launch must keep
 * running from the mount (logged). A copy is committed by renaming a finished, verified partial
 * directory, so a crashed or concurrent copy is never used.
 */
export async function preparePayloadCopy(host: PayloadCopyHost): Promise<string | null> {
  const { root, key } = host.plan
  const build: string = path.join(root, key)

  if (await holdPayloadCopy(build, key)) {
    const now = new Date()
    await fs.promises.utimes(path.join(build, MARKER), now, now).catch(() => undefined)

    return build
  }

  const partial: string = path.join(root, `${PARTIAL_PREFIX}${host.pid ?? process.pid}-${randomUUID()}`)

  try {
    if (fs.existsSync(build)) {
      throw new Error(`${build} exists without a completion marker; remove it to let Hermes copy the payload again`)
    }

    await fs.promises.mkdir(partial, { recursive: true })
    await copyBuild(host, build, partial)

    if (!(await holdPayloadCopy(build, key))) {
      throw new Error(`${build} was removed as soon as it was copied`)
    }

    return build
  } catch (error) {
    host.log(
      `[payload] running from the AppImage mount: ${String(error)}. CLI links and the gateway stop working when the app quits.`
    )
    await fs.promises.rm(partial, { recursive: true, force: true }).catch(() => undefined)

    return null
  }
}

/** The resources directory this launch runs the agent payload from. */
export async function runnablePayloadResources(host: RunnableResourcesHost): Promise<string> {
  // An APPIMAGE inherited from a terminal or IDE AppImage is not ours to unpack.
  if (host.platform !== 'linux' || !ownAppImage(host.execPath, host.env) || host.stamp?.payload !== 'bundled') {
    return host.resourcesPath
  }

  const plan: PayloadCopyPlan | null = payloadCopyPlan(host, await host.readMountinfo().catch(() => ''))

  if (!plan) {
    return host.resourcesPath
  }

  const build: string | null = await preparePayloadCopy({ ...host, plan })

  if (!build) {
    return host.resourcesPath
  }

  void collectStalePayloadCopies({ ...host, plan }, true).catch(error =>
    host.log(`[payload] stale payload copy cleanup failed: ${String(error)}`)
  )

  return build
}

export type StablePayloadResources = (log: (line: string) => void, progress?: BootProgress) => Promise<string>

/**
 * Single-flight for the process: the first caller's progress reporter sees the one-time copy. A copy
 * that is gone by a later call (removed by hand, or by a cleanup that ran before this launch held it)
 * is prepared again instead of handed out.
 */
export function createStablePayloadResources(
  hostFor: (log: (line: string) => void, progress?: BootProgress) => RunnableResourcesHost
): StablePayloadResources {
  let runnable: Promise<string> | null = null

  return async (log, progress) => {
    const host: RunnableResourcesHost = hostFor(log, progress)
    const pending: Promise<string> = (runnable ??= runnablePayloadResources(host))
    const resources: string = await pending

    if (resources === host.resourcesPath || (await markedComplete(resources, path.basename(resources)))) {
      return resources
    }

    log(`[payload] ${resources} is gone; preparing this build's copy again`)

    if (runnable === pending) {
      runnable = null
    }

    return (runnable ??= runnablePayloadResources(host))
  }
}

export const stablePayloadResources: StablePayloadResources = createStablePayloadResources((log, progress) => ({
  platform: process.platform,
  execPath: process.execPath,
  env: process.env,
  home: os.homedir(),
  resourcesPath: process.resourcesPath,
  stamp: INSTALL_STAMP,
  readMountinfo: () => fs.promises.readFile('/proc/self/mountinfo', 'utf8'),
  binDir: path.join(os.homedir(), '.local', 'bin'),
  serviceUnitDirs: gatewayUnitDirs(process.env, os.homedir()),
  log,
  progress
}))
