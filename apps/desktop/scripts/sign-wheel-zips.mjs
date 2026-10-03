// sign-wheel-zips.mjs — sign Mach-O members inside payload uv-cache wheel zips.
//
// uv's cache keeps built wheels as ZIPs beside the extracted archive buckets.
// electron-osx-sign signs the extracted trees but cannot reach inside a ZIP;
// Apple's notary extracts every zip and validates each Mach-O, so an
// unsigned member fails the whole submission (err 4000, "not signed with a
// valid Developer ID certificate"). After our round-5 cache work the zips
// are load-bearing for offline per-install venv rebuilds, so we sign the
// members instead of dropping them.
//
// Mechanism (proven on a real macOS 26 host): extract → codesign --timestamp
// --options runtime → repack → extract again → codesign --verify --strict.
// The verify gate is mandatory: with a real Developer ID the signature is
// embedded in the Mach-O and survives plain-zip roundtrips; if a host ever
// degrades to xattr-only signatures, the gate fails the build instead of
// shipping an app the notary will reject anyway.

import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { isMachO } from './sign-nested-chromium.mjs'

const MACHO_EXTENSIONS = new Set(['.so', '.dylib'])

function listWheelFiles(payload) {
  const cache = path.join(payload, 'uv-cache')
  if (!fs.existsSync(cache)) return []
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
      if (ent.isDirectory()) walk(p)
      else if (ent.isFile() && p.endsWith('.whl')) out.push(p)
    }
  }
  walk(cache)
  return out
}

function run(exec, command, args, options = {}) {
  return exec(command, args, { stdio: ['ignore', 'pipe', 'pipe'], encoding: 'utf8', ...options })
}

export function signWheelZipMembers(payload, opts = {}) {
  const identity = opts.identity
  const wheels = listWheelFiles(payload)
  if (!identity) return { wheels: wheels.length, signed: 0, failed: 0 }
  const exec = opts.exec ?? execFileSync
  const keychain = opts.keychain || null
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'wheel-sign-'))
  try {
    let signed = 0
    let failed = 0
    for (const wheel of wheels) {
      const wheelWork = path.join(work, path.basename(wheel).replace(/[^\w.-]/g, '_'))
      fs.mkdirSync(path.join(wheelWork, 'extract'), { recursive: true })
      const extract = path.join(wheelWork, 'extract')
      run(exec, 'unzip', ['-q', '-o', wheel, '-d', extract])
      let wheelTouched = false
      const members = []
      const walk = dir => {
        for (const ent of fs.readdirSync(dir, { withFileTypes: true })) {
          const p = path.join(dir, ent.name)
          if (ent.isDirectory()) { walk(p); continue }
          if (!ent.isFile() || !MACHO_EXTENSIONS.has(path.extname(p))) continue
          members.push(p)
        }
      }
      walk(extract)
      for (const member of members) {
        if (!isMachO(member)) continue
        const args = ['--force', '--sign', identity, '--timestamp', '--options', 'runtime']
        if (keychain) args.push('--keychain', keychain)
        args.push(member)
        run(exec, 'codesign', args)
        signed += 1
        wheelTouched = true
      }
      if (!wheelTouched) {
        fs.rmSync(wheelWork, { recursive: true, force: true })
        continue
      }
      // Repack preserving directory structure; the verify gate below proves
      // the embedded signature survived this host's zip semantics.
      const repacked = path.join(wheelWork, 'repacked.whl')
      run(exec, 'zip', ['-q', '-r', '-X', repacked, '.'], { cwd: extract })
      // Verify from a FRESH extraction: the predicate the notary applies.
      const verify = path.join(wheelWork, 'verify')
      fs.mkdirSync(verify, { recursive: true })
      run(exec, 'unzip', ['-q', '-o', repacked, '-d', verify])
      for (const member of members) {
        const rel = path.relative(extract, member)
        const candidate = path.join(verify, rel)
        try {
          run(exec, 'codesign', ['--verify', '--strict', candidate])
        } catch (error) {
          failed += 1
          throw new Error(
            `sign-wheel-zips: ${path.basename(wheel)}:${rel} lost its signature through the repack ` +
            `(embedded Developer ID signatures must survive a zip roundtrip): ${error.message}`)
        }
      }
      fs.copyFileSync(repacked, wheel)
      fs.rmSync(wheelWork, { recursive: true, force: true })
    }
    return { wheels: wheels.length, signed, failed }
  } finally {
    fs.rmSync(work, { recursive: true, force: true })
  }
}
