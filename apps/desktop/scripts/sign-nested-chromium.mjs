// Sign payload Chromium with our Developer ID before osx-sign seals
// Hermes.app. Google's leftover signatures have no timestamp and no
// hardened runtime, so Apple's notary rejects them.
//
// Do not codesign a file that lives inside a .framework: codesign then
// treats Versions/A/Foo as a bundle and dies with "code object is not
// signed at all" (or "bundle format is ambiguous" on the Foo.framework/Foo
// symlink). Sign the enclosing .app with --deep. Loose Mach-O outside
// any .app (headless-shell + its dylibs) is signed as a file.
//
// signIgnore still keeps osx-sign off the chromium trees.

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

const MACHO_MAGICS = new Set([
  0xfeedface, 0xcefaedfe, 0xfeedfacf, 0xcffaedfe, 0xcafebabe, 0xbebafeca
])

export function isMachO(file) {
  const buf = Buffer.alloc(4)
  let fd
  try {
    fd = fs.openSync(file, 'r')
    if (fs.readSync(fd, buf, 0, 4, 0) !== 4) return false
  } catch {
    return false
  } finally {
    if (fd != null) fs.closeSync(fd)
  }
  return MACHO_MAGICS.has(buf.readUInt32BE(0))
}

export function chromiumRoots(payload) {
  const tools = path.join(payload, 'tools')
  if (!fs.existsSync(tools)) return []
  return fs
    .readdirSync(tools, { withFileTypes: true })
    .filter(ent => ent.isDirectory() && /^chromium(_headless_shell)?-\d+/.test(ent.name))
    .map(ent => path.join(tools, ent.name))
}

export function listTopLevelApps(root) {
  const apps = []
  const walk = dir => {
    let entries
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const ent of entries) {
      const p = path.join(dir, ent.name)
      if (ent.isSymbolicLink()) continue
      if (ent.isDirectory() && p.endsWith('.app')) {
        apps.push(p)
        continue
      }
      if (ent.isDirectory()) walk(p)
    }
  }
  walk(root)
  return apps
}

export function listLooseMachO(root) {
  const out = []
  const walk = dir => {
    let entries
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const ent of entries) {
      const p = path.join(dir, ent.name)
      if (ent.isSymbolicLink()) continue
      if (ent.isDirectory()) {
        if (p.endsWith('.app')) continue
        walk(p)
        continue
      }
      if (!ent.isFile()) continue
      if (isMachO(p)) out.push(p)
    }
  }
  walk(root)
  return out
}

// If Foo.framework/Foo is a regular file and Versions/*/Foo exists,
// replace the top-level copy with a symlink. Same for Versions/Current.
export function repairFrameworkLinks(root) {
  let repaired = 0
  const walk = dir => {
    let entries
    try {
      entries = fs.readdirSync(dir, { withFileTypes: true })
    } catch {
      return
    }
    for (const ent of entries) {
      const p = path.join(dir, ent.name)
      if (ent.isSymbolicLink()) continue
      if (ent.isDirectory() && ent.name.endsWith('.framework')) {
        repaired += repairOneFramework(p)
        continue
      }
      if (ent.isDirectory()) walk(p)
    }
  }
  walk(root)
  return repaired
}

function repairOneFramework(frameworkDir) {
  let repaired = 0
  const versions = path.join(frameworkDir, 'Versions')
  if (!fs.existsSync(versions)) return 0
  const versionDirs = fs
    .readdirSync(versions, { withFileTypes: true })
    .filter(ent => ent.isDirectory() && ent.name !== 'Current')
    .map(ent => ent.name)
  if (versionDirs.length === 0) return 0
  const preferred = versionDirs.includes('A') ? 'A' : versionDirs[0]
  const current = path.join(versions, 'Current')
  if (!fs.existsSync(current) || !fs.lstatSync(current).isSymbolicLink()) {
    if (fs.existsSync(current)) fs.rmSync(current, { recursive: true, force: true })
    fs.symlinkSync(preferred, current)
    repaired += 1
  }
  const stem = path.basename(frameworkDir, '.framework')
  for (const name of [stem, 'Resources', 'Libraries', 'Helpers']) {
    const top = path.join(frameworkDir, name)
    const target = path.posix.join('Versions', 'Current', name)
    const versioned = path.join(frameworkDir, 'Versions', preferred, name)
    if (!fs.existsSync(versioned) && name !== stem) continue
    if (fs.existsSync(top) && fs.lstatSync(top).isSymbolicLink()) continue
    if (fs.existsSync(top)) fs.rmSync(top, { recursive: true, force: true })
    if (fs.existsSync(versioned) || name === stem) {
      fs.symlinkSync(target, top)
      repaired += 1
    }
  }
  return repaired
}

export function parseDeveloperId(identityList) {
  const line = String(identityList || '')
    .split(/\r?\n/)
    .find(l => l.includes('Developer ID Application:'))
  if (!line) return null
  const quoted = line.match(/"([^"]+)"/)
  return quoted ? quoted[1] : null
}

export function findDeveloperId(exec = execFileSync, keychain = null) {
  if (process.env.CSC_NAME) return process.env.CSC_NAME
  const args = ['find-identity', '-v', '-p', 'codesigning']
  if (keychain) args.push(keychain)
  let out
  try {
    out = exec('security', args, { encoding: 'utf8' })
  } catch {
    return null
  }
  return parseDeveloperId(out)
}

export async function resolveSigningIdentity(packager, exec = execFileSync) {
  if (!packager?.codeSigningInfo?.value) {
    return { identity: findDeveloperId(exec), keychain: null }
  }
  let info
  try {
    info = await packager.codeSigningInfo.value
  } catch {
    return { identity: findDeveloperId(exec), keychain: null }
  }
  const keychain = info?.keychainFile || process.env.CSC_KEYCHAIN || null
  return { identity: findDeveloperId(exec, keychain), keychain }
}

function codesign(exec, identity, entitlements, keychain, target, deep) {
  const args = [
    '--force',
    '--sign',
    identity,
    '--timestamp',
    '--options',
    'runtime',
    '--entitlements',
    entitlements
  ]
  if (deep) args.push('--deep')
  if (keychain) args.push('--keychain', keychain)
  args.push(target)
  exec('codesign', args, { stdio: 'pipe' })
}

export function signNestedChromium(payload, opts = {}) {
  const identity = opts.identity
  if (!identity) return { signed: 0, repaired: 0, identity: null }
  const entitlements = opts.entitlements
  if (!entitlements || !fs.existsSync(entitlements)) {
    throw new Error(`sign-nested-chromium: entitlements missing at ${entitlements}`)
  }
  const exec = opts.exec ?? execFileSync
  const keychain = opts.keychain || null
  let signed = 0
  let repaired = 0
  for (const root of chromiumRoots(payload)) {
    repaired += repairFrameworkLinks(root)
    for (const app of listTopLevelApps(root)) {
      codesign(exec, identity, entitlements, keychain, app, true)
      signed += 1
    }
    for (const file of listLooseMachO(root)) {
      codesign(exec, identity, entitlements, keychain, file, false)
      signed += 1
    }
  }
  return { signed, repaired, identity }
}
