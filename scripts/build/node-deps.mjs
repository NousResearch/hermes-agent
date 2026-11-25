#!/usr/bin/env node
import { execFileSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { constants as osConstants } from 'node:os'
import { basename, delimiter, dirname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { parseArgs } from 'node:util'

import { prepareElectronRuntime } from './electron-runtime.mjs'
import { JOB_SIGNALS, lockCheckout } from './node-deps-lock.mjs'
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

// The reuse check reads the trees: an error from a run that lost its lock meanwhile (the new owner
// moved them aside) is the lost lock, not a missing file.
function reusable(lock, paths) {
  try {
    return completedInstallMatches(paths)
  } catch (error) {
    lock.check()
    throw error
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
function previousTrees(source, check) {
  const cache = join(source, '.cache')
  const aside = join(cache, 'node-deps-previous')
  const record = join(aside, 'trees.json')
  const complete = join(aside, 'complete')
  // pnpm writes this last, and only when an install succeeds.
  const finished = join(source, 'node_modules/.pnpm-workspace-state-v1.json')
  // Renamed away first: a delete killed halfway must never leave a partial set
  // under the name restore reads.
  // Every step that changes the trees first checks that this run still holds the lock.
  const discard = () => {
    check()
    if (existsSync(aside)) renameSync(aside, join(cache, `node-deps-discard-${process.pid}-${Date.now()}`))
    for (const name of existsSync(cache) ? readdirSync(cache) : []) {
      if (!name.startsWith('node-deps-discard-')) continue
      check()
      rmSync(join(cache, name), { recursive: true, force: true, maxRetries: 3 })
    }
  }
  const tree = project => join(source, project, 'node_modules')
  // `complete` goes first, so a restore killed halfway is resumed, never mistaken
  // for a set a finished install replaced. Each step can run again.
  const restore = () => {
    check()
    rmSync(complete, { force: true })
    const { moved = [], absent = [] } = existsSync(record) ? JSON.parse(readFileSync(record, 'utf8')) : {}
    for (const project of absent) {
      check()
      rmSync(tree(project), { recursive: true, force: true, maxRetries: 3 })
    }
    for (const project of moved) {
      const kept = join(aside, project, 'node_modules')
      // Gone: already put back. A workspace the checkout no longer has (a pull
      // dropped it) is discarded with the set.
      if (!existsSync(kept) || !existsSync(join(source, project, 'package.json'))) continue
      check()
      rmSync(tree(project), { recursive: true, force: true, maxRetries: 3 })
      check()
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
    check()
    mkdirSync(aside, { recursive: true })
    // Renamed into place: a run killed (or out of space) mid-write must never leave a record
    // that every later run fails to parse.
    writeFileSync(`${record}.tmp`, JSON.stringify({ moved, absent: projects.filter(project => !moved.includes(project)) }))
    renameSync(`${record}.tmp`, record)
    try {
      for (const project of moved) {
        check()
        mkdirSync(join(aside, project), { recursive: true })
        moveTree(tree(project), join(aside, project, 'node_modules'))
      }
      check()
      writeFileSync(complete, '')
    } catch (error) {
      if (error.code !== 'ENODEPSLOCK') restore()
      throw error
    }
  }
  return { recover, restore, discard, moveAside }
}

// A node_modules that is a mount point (a container volume) cannot be renamed (EBUSY),
// nor one whose .cache is another filesystem (EXDEV): it is emptied in place, as before
// #150031, and only that tree cannot come back. On Windows EBUSY is an open file; deleting
// around it would gut the tree, so the move fails and the set is restored.
function moveTree(from, to) {
  try {
    renameSync(from, to)
  } catch (error) {
    if (error.code !== 'EXDEV' && !(error.code === 'EBUSY' && process.platform !== 'win32')) throw error
    console.warn(`node-deps: cannot move ${from} aside (${error.code}); deleting it, so a failed install cannot restore it`)
    for (const entry of readdirSync(from)) rmSync(join(from, entry), { recursive: true, force: true, maxRetries: 3 })
  }
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

// What this run installs and the key its receipt carries, read once the run holds the lock: an
// update that changed them while this run waited is installed, not judged against older files.
function dependencyInputs({ source, workspaces, env, nativeToolchain, electron }) {
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
  const inputs = createHash('sha256').update(JSON.stringify({
    node: process.versions.node, pnpm: version, platform: process.platform, arch: process.arch, args,
    // Windows Python uppercases inherited keys. Cache, store and state locations
    // change where pnpm keeps data, and the retry policy how it downloads, not what it installs.
    config: Object.entries(env).filter(([key]) => /^p?npm_config_/i.test(key)
      && !/^p?npm_config_(cache|cache_dir|store_dir|state_dir|offline|prefer_offline|fetch_retries|fetch_retry_\w+)$/i.test(key))
      .map(([key, value]) => [key.toLowerCase(), value]).sort(),
  }))
  const files = ['pnpm-lock.yaml', 'pnpm-workspace.yaml', '.npmrc', ...projects.map(path => join(path, 'package.json'))].sort()
  for (const file of files) {
    inputs.update(file).update('\0').update(existsSync(join(source, file)) ? readFileSync(join(source, file)) : '<missing>').update('\0')
  }
  const nativeKey = nativeToolchain === undefined ? undefined : JSON.stringify(nativeToolchain)
  return { lockText, projects, selected, prepareElectron, pnpm, args, key: inputs.digest('hex'), nativeKey }
}

/** Install the full requested workspace union in one strict, locked operation.
 *  ``electron: false`` leaves Desktop's Electron binary unprepared, for builds that never run it. */
export function prepareNodeDependencies({ source, workspaces, env = process.env, reuse = false, install = true, nativeToolchain, electron = true }) {
  source = resolve(source)
  if (!Array.isArray(workspaces) || workspaces.length === 0) {
    throw new Error('Select at least one workspace; implicit all-workspace installation is not allowed')
  }
  // This receipt certifies dependency preparation, never compiled product freshness.
  // Keep it inside the cached tree so removing node_modules also removes the receipt.
  const receipt = join(source, 'node_modules/.hermes-node-deps')
  // Ordinary product builders consume the baseline receipt; preparation also
  // binds lifecycle outputs to its compiler/SDK identity.
  const nativeReceipt = `${receipt}.native-toolchain`
  // pnpm's copy of the lockfile it last installed.
  const installedLock = join(source, 'node_modules/.pnpm/lock.yaml')
  const lock = lockCheckout(source)
  let selected
  try {
    const inputs = dependencyInputs({ source, workspaces, env, nativeToolchain, electron })
    const { lockText, projects, prepareElectron, pnpm, args, key, nativeKey } = inputs
    selected = inputs.selected
    const previous = previousTrees(source, lock.check)
    previous.recover()
    if (reuse && reusable(lock, { source, receipt, installedLock, key, nativeKey })) {
      lock.check()
      if (prepareElectron) prepareElectronRuntime({ source, install, env, run: lock.run })
      console.log(`node-deps: reusing completed install (${selected.join(', ')})`)
      lock.release()
      return { source, workspaces: selected }
    }
    if (!install) throw new Error('Workspace dependencies are stale or missing and lazy installs are disabled; run an explicit build/update')
    // The receipts live inside the trees, so they leave and return with them.
    previous.moveAside(projects)
    try {
      console.log(`node-deps: installing workspace dependencies with pnpm (${selected.join(', ')})...`)
      lock.check()
      lock.run(pnpm, args, { cwd: source, env })
      if (!installedLockMatches(lockText, installedLock, selected)) {
        throw new Error(`pnpm did not record the locked install of ${selected.join(', ')} in ${installedLock}`)
      }
      lock.check()
      if (prepareElectron) prepareElectronRuntime({ source, install, env, run: lock.run })
    } catch (error) {
      // A run that lost the lock leaves the trees, and the set aside, to the run that holds it.
      if (error.code === 'ENODEPSLOCK') throw error
      previous.restore()
      console.error('node-deps: install failed; restored the previously installed dependencies')
      throw error
    }
    if (reuse) {
      const completed = `${key}\n${createHash('sha256').update(readFileSync(installedLock)).digest('hex')}\n`
      lock.check()
      writeFileSync(receipt, completed)
      if (nativeKey !== undefined) {
        lock.check()
        writeFileSync(nativeReceipt, `${completed}${nativeKey}\n`)
      }
    }
    // The install succeeded: a set that cannot be deleted now (an antivirus or indexer handle on
    // Windows) is discarded by the next run's recovery instead of failing this one.
    try {
      previous.discard()
    } catch (error) {
      if (error.code === 'ENODEPSLOCK') throw error
      console.warn(`node-deps: could not delete the previous dependencies yet (${error.code ?? error.message}); the next run removes them`)
    }
  } catch (error) {
    // A release that fails too (a leaked guard) must not hide why the run failed.
    try {
      lock.release()
    } catch (release) {
      console.warn(`node-deps: could not release the checkout lock (${release.message})`)
    }
    throw error
  }
  lock.release()
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
  try {
    prepareNodeDependencies({ source: values.source, workspaces: values.workspace, reuse: values.reuse, install: !values['no-install'], nativeToolchain: values['native-toolchain'] })
  } catch (error) {
    // A child ended by a job signal this run outlived (the relay's 128 + n): exit as that signal
    // would have, 130 for Ctrl-C, now that the trees are back.
    const interrupted = JOB_SIGNALS.map(name => 128 + osConstants.signals[name]).includes(error.status)
    if (!interrupted) throw error
    console.error(`node-deps: ${error.message} (exit ${error.status})`)
    process.exit(error.status)
  }
}
