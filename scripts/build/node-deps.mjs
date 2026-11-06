#!/usr/bin/env node
import { execFileSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import {
  closeSync, existsSync, linkSync, mkdirSync, openSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, statSync,
  writeFileSync, writeSync,
} from 'node:fs'
import { hostname } from 'node:os'
import { basename, delimiter, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

import { prepareElectronRuntime } from './electron-runtime.mjs'
import { lockImporters } from './pnpm-lock.mjs'

// pnpm 12 is a native executable (pnpm.exe on Windows, never a .cmd shim), so it
// runs without a shell and paths stay argv. Nix's pnpm is a shell wrapper that
// execFile runs through its shebang.
export function pnpmCommand({ env = process.env } = {}) {
  const name = process.platform === 'win32' ? 'pnpm.exe' : 'pnpm'
  for (const dir of (env.PATH || env.Path || '').split(delimiter)) {
    const bin = dir && join(dir, name)
    if (bin && existsSync(bin)) return bin
  }
  // Under `pnpm run`, the pnpm running this script: it was started by path, not from PATH.
  const running = env.npm_execpath
  if (running && basename(running) === name && existsSync(running)) return running
  throw new Error('pnpm is required on PATH')
}

// The native package ships its manifest beside the executable: no child spawn,
// which runs before the reuse short-circuit. Layouts without one (Nix) fall back
// to asking pnpm.
function pnpmVersion(pnpm, { source, env }) {
  try {
    return JSON.parse(readFileSync(join(dirname(realpathSync(pnpm)), 'package.json'), 'utf8')).version
  } catch {
    return execFileSync(pnpm, ['--version'], { cwd: source, env, encoding: 'utf8' }).trim()
  }
}

// PM's pnpm pin: the pnpm this checkout's lockfile is written and installed with.
function pinnedPnpm(source) {
  const path = join(source, 'pm/lock.json')
  let version
  try {
    version = JSON.parse(readFileSync(path, 'utf8')).packages?.pnpm?.version
  } catch (error) {
    throw new Error(`cannot read the pnpm pin from ${path}: ${error.message}`)
  }
  if (typeof version !== 'string') throw new Error(`${path} pins no pnpm (packages.pnpm.version)`)
  return version
}

// Every package directory pnpm linked: each locked direct dependency of an installed
// project, and every entry of every virtual-store slot (the package itself plus links
// to its dependencies, so a removed slot shows up as a dangling link). Optional
// dependencies are skipped on hosts they do not support.
function linkedPackageDirs(source, installedLock) {
  const dirs = []
  for (const [project, fields] of Object.entries(lockImporters(readFileSync(installedLock, 'utf8')))) {
    for (const name of Object.keys({ ...fields.dependencies, ...fields.devDependencies })) {
      dirs.push(join(source, project, 'node_modules', name))
    }
  }
  const virtualStore = join(source, 'node_modules/.pnpm')
  for (const slot of readdirSync(virtualStore, { withFileTypes: true })) {
    if (!slot.isDirectory() || slot.name === 'node_modules') continue
    const modules = join(virtualStore, slot.name, 'node_modules')
    for (const entry of existsSync(modules) ? readdirSync(modules, { withFileTypes: true }) : []) {
      const path = join(modules, entry.name)
      if (entry.name.startsWith('@') && entry.isDirectory()) {
        dirs.push(...readdirSync(path).map(scoped => join(path, scoped)))
      } else {
        dirs.push(path)
      }
    }
  }
  return dirs
}

function completedInstallMatches({ source, receipt, installedLock, key, nativeKey }) {
  if (!existsSync(receipt) || !existsSync(installedLock)) return false
  const expected = `${key}\n${createHash('sha256').update(readFileSync(installedLock)).digest('hex')}\n`
  if (nativeKey !== undefined) {
    const nativeReceipt = `${receipt}.native-toolchain`
    if (!existsSync(nativeReceipt) || readFileSync(nativeReceipt, 'utf8') !== `${expected}${nativeKey}\n`) return false
  }
  // Probe each installed package's manifest, not its directory: a partial cleanup
  // or snapshot restore can leave gutted directories behind, and product builders
  // resolve tools through these manifests (#128935).
  return readFileSync(receipt, 'utf8') === expected
    && linkedPackageDirs(source, installedLock).every(dir => existsSync(join(dir, 'package.json')))
}

// One run at a time per checkout. The callers hold different locks or none (`hermes update`,
// `hermes --tui`, the dashboard, Desktop, another profile), and a run moves trees in and out of
// place, so even a reuse check must not run beside an install. The lock names its owner: a
// killed run's lock is taken over (and its trees then recovered), a live one is waited for.
const sleep = ms => Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms)

function readText(path) {
  try {
    return readFileSync(path, 'utf8')
  } catch (error) {
    if (error.code === 'ENOENT') return undefined
    throw error
  }
}

// Linux's process start time, so a dead owner's reused pid does not hold the lock forever.
function startTime(pid) {
  try {
    const stat = readFileSync(`/proc/${pid}/stat`, 'utf8')
    return stat.slice(stat.lastIndexOf(')') + 2).split(' ')[19]
  } catch {
    return undefined
  }
}

function ownerAlive(held, lock) {
  let owner
  try {
    owner = JSON.parse(held)
  } catch {
    // Created but not yet written, or a writer killed in between.
    try {
      return Date.now() - statSync(lock).mtimeMs < 10_000
    } catch {
      return false
    }
  }
  // Another machine's run on a shared checkout cannot be judged from here.
  if (owner.host !== hostname()) return true
  try {
    process.kill(owner.pid, 0)
  } catch (error) {
    return error.code === 'EPERM'
  }
  return owner.start === undefined || owner.start === startTime(owner.pid)
}

function lockCheckout(source) {
  const lock = join(source, '.cache/node-deps.lock')
  const self = JSON.stringify({ pid: process.pid, host: hostname(), start: startTime(process.pid) })
  let announced = false
  for (;;) {
    let fd
    try {
      mkdirSync(dirname(lock), { recursive: true })
      fd = openSync(lock, 'wx')
    } catch (error) {
      // No run can write a checkout this process cannot write, so a reader needs no lock.
      if (['EACCES', 'EPERM', 'EROFS'].includes(error.code)) return () => {}
      if (error.code !== 'EEXIST') throw error
    }
    if (fd !== undefined) {
      try {
        writeSync(fd, self)
      } catch (error) {
        rmSync(lock, { force: true })
        throw error
      } finally {
        closeSync(fd)
      }
      return () => { if (readText(lock) === self) rmSync(lock, { force: true }) }
    }
    const held = readText(lock)
    if (held === undefined) continue
    if (!ownerAlive(held, lock)) {
      // Another waiter may have taken the dead lock over first: what was renamed is then
      // its live lock, which goes back unless a third run has locked in the meantime.
      const reap = `${lock}.${process.pid}.stale`
      try {
        renameSync(lock, reap)
      } catch (error) {
        if (error.code === 'ENOENT') continue
        throw error
      }
      if (readText(reap) !== held) {
        try {
          linkSync(reap, lock)
        } catch (error) {
          if (error.code !== 'EEXIST') throw error
        }
      }
      rmSync(reap, { force: true })
      continue
    }
    if (!announced) console.log(`node-deps: waiting for another dependency install in this checkout (${held}); if none is running, delete ${lock}`)
    announced = true
    sleep(500)
  }
}

// pnpm's frozen install is incremental: it trusts its record of what is installed,
// so it neither restores a gutted package nor reruns dependency builds for a changed
// toolchain or configuration. Every install starts from empty trees, as npm ci did;
// the store keeps that local. This also removes an npm-made tree, which pnpm would
// move aside and leave behind.
// The previous trees are moved aside, not deleted, until the install succeeds: the
// first pnpm install runs on an empty store, so an unreachable registry would
// otherwise leave no node_modules at all (#150031). Both generations are on disk
// while pnpm runs. `pnpm fetch` would download every workspace (it takes no filter).
// .cache/node-deps-previous holds trees.json (written first: the projects whose trees
// move, and those with none, whose partial trees a restore removes), each
// <project>/node_modules, and `complete` once every tree has moved.
function previousTrees(source) {
  const cache = join(source, '.cache')
  const aside = join(cache, 'node-deps-previous')
  const record = join(aside, 'trees.json')
  const complete = join(aside, 'complete')
  // pnpm writes this last, and only when an install succeeds.
  const finished = join(source, 'node_modules/.pnpm-workspace-state-v1.json')
  // Renamed away first: a delete killed halfway must never leave a partial set
  // under the name restore reads.
  const discard = () => {
    if (existsSync(aside)) renameSync(aside, join(cache, `node-deps-discard-${process.pid}-${Date.now()}`))
    for (const name of existsSync(cache) ? readdirSync(cache) : []) {
      if (name.startsWith('node-deps-discard-')) rmSync(join(cache, name), { recursive: true, force: true, maxRetries: 3 })
    }
  }
  const tree = project => join(source, project, 'node_modules')
  // `complete` goes first, so a restore killed halfway is resumed, never mistaken
  // for a set a finished install replaced. Each step can run again.
  const restore = () => {
    rmSync(complete, { force: true })
    const { moved = [], absent = [] } = existsSync(record) ? JSON.parse(readFileSync(record, 'utf8')) : {}
    for (const project of absent) rmSync(tree(project), { recursive: true, force: true, maxRetries: 3 })
    for (const project of moved) {
      const kept = join(aside, project, 'node_modules')
      // Gone: already put back. A workspace the checkout no longer has (a pull
      // dropped it) is discarded with the set.
      if (!existsSync(kept) || !existsSync(join(source, project, 'package.json'))) continue
      rmSync(tree(project), { recursive: true, force: true, maxRetries: 3 })
      renameSync(kept, tree(project))
    }
    discard()
  }
  // A set left aside means a run was killed. Once every tree had moved, a pnpm
  // install that finished since (that run's, or a manual one) is newer than the set
  // and stays; otherwise the move or a restore was cut short and is completed.
  const recover = () => (existsSync(complete) && existsSync(finished) ? discard() : restore())
  const moveAside = projects => {
    const moved = projects.filter(project => existsSync(tree(project)))
    mkdirSync(aside, { recursive: true })
    writeFileSync(record, JSON.stringify({ moved, absent: projects.filter(project => !moved.includes(project)) }))
    try {
      for (const project of moved) {
        mkdirSync(join(aside, project), { recursive: true })
        renameSync(tree(project), join(aside, project, 'node_modules'))
      }
      writeFileSync(complete, '')
    } catch (error) {
      restore()
      throw error
    }
  }
  return { recover, restore, discard, moveAside }
}

// pnpm's record of the trees it just wrote (node_modules/.pnpm/lock.yaml) must be the source
// lockfile's for the selected projects and every other project it installed: no other run's.
// An optional dependency this host skips is left out of the record, never changed.
function installedLockMatches(lockText, installedLock, selected) {
  if (!existsSync(installedLock)) return false
  const wanted = lockImporters(lockText)
  const installed = lockImporters(readFileSync(installedLock, 'utf8'))
  const same = (a = {}, b = {}) => JSON.stringify(a) === JSON.stringify(b)
  return ['.', ...selected].every(project => project in installed)
    && Object.entries(installed).every(([project, fields]) => project in wanted
      && same(fields.dependencies, wanted[project].dependencies)
      && same(fields.devDependencies, wanted[project].devDependencies)
      && Object.entries(fields.optionalDependencies ?? {})
        .every(([name, entry]) => same(entry, wanted[project].optionalDependencies?.[name])))
}

/** Install the full requested workspace union in one strict, locked operation.
 *  ``electron: false`` leaves Desktop's Electron binary unprepared, for builds that never run it. */
export function prepareNodeDependencies({ source, workspaces, env = process.env, reuse = false, install = true, nativeToolchain, electron = true }) {
  source = resolve(source)
  if (!Array.isArray(workspaces) || workspaces.length === 0) {
    throw new Error('Select at least one workspace; implicit all-workspace installation is not allowed')
  }
  const lockText = readFileSync(join(source, 'pnpm-lock.yaml'), 'utf8')
  const projects = Object.keys(lockImporters(lockText)).sort()
  const selected = [...new Set(workspaces.map(workspace => {
    // pnpm ignores a filter that matches nothing, so an unknown path must fail here.
    if (workspace === '.' || !projects.includes(workspace) || !existsSync(join(source, workspace, 'package.json'))) {
      throw new Error(`Unknown or missing locked workspace: ${workspace}`)
    }
    return workspace
  }))].sort()
  const prepareElectron = electron && selected.includes('apps/desktop')
    && Boolean(JSON.parse(readFileSync(join(source, 'apps/desktop/package.json'), 'utf8')).devDependencies?.electron)
  const pnpm = pnpmCommand({ env })
  const version = pnpmVersion(pnpm, { source, env })
  // pnpm never checks its own version here (pmOnFail: ignore keeps the lock one YAML
  // document): PM's pin is the one this checkout's lockfile was written with.
  const pinned = pinnedPnpm(source)
  if (pinned !== version) {
    throw new Error(`pnpm ${pinned} is required (pm/lock.json), found ${version} at ${pnpm}`)
  }
  // The root project is always installed with a filtered install. The release-age
  // policy is enforced where the lockfile is written (CI), not re-fetched per user.
  const args = ['install', '--frozen-lockfile', '--trust-lockfile', '--prod=false',
    ...selected.flatMap(workspace => ['--filter', `{./${workspace}}...`]),
  ]
  // This receipt certifies dependency preparation, never compiled product freshness.
  // Keep it inside the cached tree so removing node_modules also removes the receipt.
  const receipt = join(source, 'node_modules/.hermes-node-deps')
  // Ordinary product builders consume the baseline receipt; preparation also
  // binds lifecycle outputs to its compiler/SDK identity.
  const nativeReceipt = `${receipt}.native-toolchain`
  const nativeKey = nativeToolchain === undefined ? undefined : JSON.stringify(nativeToolchain)
  // pnpm's copy of the lockfile it last installed.
  const installedLock = join(source, 'node_modules/.pnpm/lock.yaml')
  const inputs = createHash('sha256').update(JSON.stringify({
    node: process.versions.node, pnpm: version, platform: process.platform, arch: process.arch, args,
    // Windows Python uppercases inherited keys. Cache, store and state locations
    // change where pnpm keeps data, not what it installs.
    config: Object.entries(env).filter(([key]) => /^p?npm_config_/i.test(key)
      && !/^p?npm_config_(cache|cache_dir|store_dir|state_dir|offline|prefer_offline)$/i.test(key))
      .map(([key, value]) => [key.toLowerCase(), value]).sort(),
  }))
  const files = ['pnpm-lock.yaml', 'pnpm-workspace.yaml', '.npmrc', ...projects.map(path => join(path, 'package.json'))].sort()
  for (const file of files) {
    inputs.update(file).update('\0').update(existsSync(join(source, file)) ? readFileSync(join(source, file)) : '<missing>').update('\0')
  }
  const key = inputs.digest('hex')
  const release = lockCheckout(source)
  try {
    const previous = previousTrees(source)
    previous.recover()
    if (reuse && completedInstallMatches({ source, receipt, installedLock, key, nativeKey })) {
      if (prepareElectron) prepareElectronRuntime({ source, install, env })
      console.log(`node-deps: reusing completed install (${selected.join(', ')})`)
      return { source, workspaces: selected }
    }
    if (!install) throw new Error('Workspace dependencies are stale or missing and lazy installs are disabled; run an explicit build/update')
    // The receipts live inside the trees, so they leave and return with them.
    previous.moveAside(projects)
    try {
      console.log(`node-deps: installing workspace dependencies with pnpm (${selected.join(', ')})...`)
      execFileSync(pnpm, args, { cwd: source, env, stdio: 'inherit' })
      if (!installedLockMatches(lockText, installedLock, selected)) {
        throw new Error(`pnpm did not record the locked install of ${selected.join(', ')} in ${installedLock}`)
      }
      if (prepareElectron) prepareElectronRuntime({ source, install, env })
    } catch (error) {
      previous.restore()
      console.error('node-deps: install failed; restored the previously installed dependencies')
      throw error
    }
    if (reuse) {
      const completed = `${key}\n${createHash('sha256').update(readFileSync(installedLock)).digest('hex')}\n`
      writeFileSync(receipt, completed)
      if (nativeKey !== undefined) writeFileSync(nativeReceipt, `${completed}${nativeKey}\n`)
    }
    previous.discard()
  } finally {
    release()
  }
  return { source, workspaces: selected }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  // The Python bundle driver uses this same resolver, not a second layout probe.
  if (process.argv[2] === '--pnpm') {
    const child = spawnSync(pnpmCommand(), process.argv.slice(3), { stdio: 'inherit' })
    if (child.error) throw child.error
    process.exit(child.status ?? 1)
  }
  const { values } = parseArgs({ options: {
    source: { type: 'string' }, workspace: { type: 'string', multiple: true },
    reuse: { type: 'boolean', default: false },
    'no-install': { type: 'boolean', default: false },
    'native-toolchain': { type: 'string' },
  } })
  if (!values.source) throw new Error('--source is required')
  prepareNodeDependencies({ source: values.source, workspaces: values.workspace, reuse: values.reuse, install: !values['no-install'], nativeToolchain: values['native-toolchain'] })
}
