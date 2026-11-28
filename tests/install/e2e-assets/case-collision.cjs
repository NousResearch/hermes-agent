'use strict'
const { execFileSync, spawnSync } = require('node:child_process')
const fs = require('node:fs')
const path = require('node:path')

const GIT_MAX_BUFFER = 64 * 1024 * 1024

function runGit(gitExe, repoDir, args, encoding) {
  return execFileSync(gitExe, ['-C', repoDir, ...args], {
    encoding,
    maxBuffer: GIT_MAX_BUFFER,
  })
}

function isIgnoreCase(gitExe, repoDir) {
  const result = spawnSync(gitExe, ['-C', repoDir, 'config', '--type=bool', '--get', 'core.ignorecase'], {
    encoding: 'utf8',
    maxBuffer: GIT_MAX_BUFFER,
  })
  if (result.error) throw result.error
  if (result.status !== 0 && result.status !== 1) {
    throw new Error(`git config core.ignorecase failed (exit ${result.status}): ${(result.stderr || '').trim()}`)
  }
  return result.status === 0 && result.stdout.trim() === 'true'
}

function parsePorcelainV2Z(raw) {
  const tokens = raw.split('\0')
  const entries = []
  for (let i = 0; i < tokens.length; i++) {
    const token = tokens[i]
    if (!token) continue
    if (token.startsWith('2 ')) {
      const orig = tokens[++i] || ''
      entries.push({ kind: '2', raw: `${token}\t${orig}` })
      continue
    }
    if (token.startsWith('1 ')) {
      const parts = []
      let pos = 0
      for (let field = 0; field < 8; field++) {
        const next = token.indexOf(' ', pos)
        if (next === -1) break
        parts.push(token.slice(pos, next))
        pos = next + 1
      }
      if (parts.length === 8) {
        entries.push({
          kind: '1',
          xy: parts[1],
          sub: parts[2],
          mH: parts[3],
          mI: parts[4],
          mW: parts[5],
          hH: parts[6],
          hI: parts[7],
          path: token.slice(pos),
          raw: token,
        })
        continue
      }
    }
    entries.push({ kind: token[0], raw: token })
  }
  return entries
}

function readHeadCollisionGroups(gitExe, repoDir) {
  const raw = runGit(gitExe, repoDir, ['ls-tree', '-r', '-z', 'HEAD'], 'utf8')
  const byFolded = new Map()
  for (const record of raw.split('\0')) {
    if (!record) continue
    const tab = record.indexOf('\t')
    if (tab === -1) continue
    const [mode, type, oid] = record.slice(0, tab).split(' ')
    if (type !== 'blob') continue
    const relPath = record.slice(tab + 1)
    const key = relPath.toLowerCase()
    let group = byFolded.get(key)
    if (!group) {
      group = []
      byFolded.set(key, group)
    }
    group.push({ mode, oid, path: relPath })
  }
  const collisions = new Map()
  for (const [key, group] of byFolded) {
    if (group.length > 1) collisions.set(key, group)
  }
  return collisions
}

function stripCrBeforeLf(buf) {
  const out = Buffer.allocUnsafe(buf.length)
  let written = 0
  for (let i = 0; i < buf.length; i++) {
    if (buf[i] === 0x0d && i + 1 < buf.length && buf[i + 1] === 0x0a) continue
    out[written++] = buf[i]
  }
  return out.subarray(0, written)
}

function matchesCollidingSibling(repoDir, gitExe, entry, group, blobCache) {
  if (!group || !group.some(item => item.path === entry.path)) return false
  const fullPath = path.join(repoDir, entry.path)
  let stat
  try {
    stat = fs.lstatSync(fullPath)
  } catch {
    return false
  }
  let diskBytes = null
  let linkTarget = null
  for (const sibling of group) {
    if (sibling.path === entry.path) continue
    if (entry.mW !== entry.mI && entry.mW !== sibling.mode) continue
    let blob = blobCache.get(sibling.oid)
    if (!blob) {
      blob = runGit(gitExe, repoDir, ['cat-file', 'blob', sibling.oid])
      blobCache.set(sibling.oid, blob)
    }
    if (sibling.mode === '120000') {
      if (!stat.isSymbolicLink()) continue
      if (linkTarget === null) linkTarget = Buffer.from(fs.readlinkSync(fullPath))
      if (linkTarget.equals(blob)) return true
      continue
    }
    if (!stat.isFile()) continue
    if (diskBytes === null) diskBytes = fs.readFileSync(fullPath)
    if (diskBytes.equals(blob)) return true
    if (!blob.includes(0) && stripCrBeforeLf(diskBytes).equals(blob)) return true
  }
  return false
}

// Historical tags (v2026.8.16.2-v2026.8.31) ship case-colliding paths that
// APFS/NTFS checkouts leave modified with a sibling blob's bytes on disk.
function inspectCaseCollisionDirty(repoDir, gitExe = process.env.HERMES_E2E_REAL_GIT || 'git') {
  const raw = runGit(gitExe, repoDir, ['-c', 'core.quotepath=false', 'status', '--porcelain=v2', '-z', '--untracked-files=all'], 'utf8')
  const entries = parsePorcelainV2Z(raw)
  if (entries.length === 0) return { ignoreCase: false, collisions: [], other: [] }
  const ignoreCase = isIgnoreCase(gitExe, repoDir)
  if (!ignoreCase) return { ignoreCase: false, collisions: [], other: entries.map(entry => entry.raw) }
  const isCandidate = entry =>
    entry.kind === '1' && entry.xy === '.M' && entry.sub === 'N...' &&
    entry.mH === entry.mI && entry.hH === entry.hI
  if (!entries.some(isCandidate)) {
    return { ignoreCase: true, collisions: [], other: entries.map(entry => entry.raw) }
  }
  const groups = readHeadCollisionGroups(gitExe, repoDir)
  const blobCache = new Map()
  const collisions = []
  const other = []
  for (const entry of entries) {
    if (isCandidate(entry) && matchesCollidingSibling(repoDir, gitExe, entry, groups.get(entry.path.toLowerCase()), blobCache)) {
      collisions.push(entry.path)
    } else {
      other.push(entry.raw)
    }
  }
  return { ignoreCase: true, collisions, other }
}

module.exports = { inspectCaseCollisionDirty, parsePorcelainV2Z }

if (require.main === module) {
  const argv = process.argv.slice(2)
  const mode = argv[0]
  const repoDir = argv[1]
  let gitExe = process.env.HERMES_E2E_REAL_GIT || 'git'
  if (argv.length === 4 && argv[2] === '--git' && argv[3]) {
    gitExe = argv[3]
  } else if (argv.length !== 2) {
    console.error('usage: node case-collision.cjs (--list | --only-collisions) <repoDir> [--git <gitExe>]')
    process.exit(2)
  }
  if (!repoDir || (mode !== '--list' && mode !== '--only-collisions')) {
    console.error('usage: node case-collision.cjs (--list | --only-collisions) <repoDir> [--git <gitExe>]')
    process.exit(2)
  }
  try {
    const result = inspectCaseCollisionDirty(repoDir, gitExe)
    if (mode === '--list') {
      console.log(JSON.stringify(result))
    } else if (result.collisions.length > 0 && result.other.length === 0) {
      console.log(`tolerating HEAD case-collision dirty path(s): ${result.collisions.join(', ')}`)
    } else {
      process.exitCode = 1
    }
  } catch (error) {
    console.error(`case-collision check failed: ${error.message}`)
    process.exitCode = 2
  }
}
