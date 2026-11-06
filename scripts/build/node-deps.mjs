#!/usr/bin/env node
import { execFileSync, spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { existsSync, readdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs'
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

// pnpm's frozen install is incremental: it trusts its record of what is installed,
// so it neither restores a gutted package nor reruns dependency builds for a changed
// toolchain or configuration. Every install starts from empty trees, as npm ci did;
// the store keeps that local. This also removes an npm-made tree, which pnpm would
// move aside and leave behind.
function removeInstalledTrees(source, projects) {
  for (const project of projects) {
    rmSync(join(source, project, 'node_modules'), { recursive: true, force: true, maxRetries: 3 })
  }
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
  // pnpm-workspace.yaml sets pmOnFail: ignore (so the lock stays one YAML document),
  // which also stops pnpm from enforcing the root devEngines pin: enforce it here.
  const pinned = JSON.parse(readFileSync(join(source, 'package.json'), 'utf8')).devEngines?.packageManager
  if (pinned?.name === 'pnpm' && pinned.version !== version) {
    throw new Error(`pnpm ${pinned.version} is required (root package.json devEngines), found ${version} at ${pnpm}`)
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
  if (reuse && completedInstallMatches({ source, receipt, installedLock, key, nativeKey })) {
    if (prepareElectron) prepareElectronRuntime({ source, install, env })
    console.log(`node-deps: reusing completed install (${selected.join(', ')})`)
    return { source, workspaces: selected }
  }
  if (!install) throw new Error('Workspace dependencies are stale or missing and lazy installs are disabled; run an explicit build/update')
  // pnpm can fail during validation before touching node_modules. Invalidate first.
  rmSync(receipt, { force: true })
  rmSync(nativeReceipt, { force: true })
  removeInstalledTrees(source, projects)
  console.log(`node-deps: installing workspace dependencies with pnpm (${selected.join(', ')})...`)
  execFileSync(pnpm, args, { cwd: source, env, stdio: 'inherit' })
  if (prepareElectron) prepareElectronRuntime({ source, install, env })
  if (reuse) {
    const completed = `${key}\n${createHash('sha256').update(readFileSync(installedLock)).digest('hex')}\n`
    writeFileSync(receipt, completed)
    if (nativeKey !== undefined) writeFileSync(nativeReceipt, `${completed}${nativeKey}\n`)
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
