/**
 * Hash the desktop source state that affects a packaged app build.
 *
 * This intentionally excludes generated build outputs such as dist/, build/,
 * and release/. The hash is written into install-stamp.json at build time and
 * compared at runtime so dirty local builds do not ask to rebuild forever.
 */

const crypto = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')
const { execFileSync } = require('node:child_process')

const DESKTOP_SOURCE_STATE_PATHS = [
  'apps/desktop/assets',
  'apps/desktop/electron',
  'apps/desktop/index.html',
  'apps/desktop/package.json',
  'apps/desktop/public',
  'apps/desktop/scripts',
  'apps/desktop/src',
  'apps/desktop/tsconfig.json',
  'apps/desktop/vite.config.ts',
  'apps/shared',
  'package-lock.json',
  'package.json'
]

function hashText(value) {
  return crypto.createHash('sha256').update(value || '', 'utf8').digest('hex')
}

function desktopSourceStateMaterial(diff, untrackedEntries) {
  return ['diff-v1', diff || '', 'untracked-v1', (untrackedEntries || []).join('\n')].join('\0')
}

const EMPTY_DESKTOP_SOURCE_STATE_HASH = hashText(desktopSourceStateMaterial('', []))

function gitOutput(repoRoot, args, options = {}) {
  try {
    return execFileSync('git', args, {
      cwd: repoRoot,
      encoding: options.encoding || 'utf8',
      maxBuffer: 64 * 1024 * 1024,
      stdio: ['ignore', 'pipe', 'ignore']
    })
  } catch {
    return null
  }
}

function hashFile(filePath, fsImpl = fs) {
  return crypto.createHash('sha256').update(fsImpl.readFileSync(filePath)).digest('hex')
}

function currentDesktopSourceStateHash(repoRoot, fsImpl = fs) {
  if (!repoRoot) return null

  const diff = gitOutput(repoRoot, ['diff', '--binary', 'HEAD', '--', ...DESKTOP_SOURCE_STATE_PATHS])
  const untrackedOutput = gitOutput(repoRoot, [
    'ls-files',
    '--others',
    '--exclude-standard',
    '-z',
    '--',
    ...DESKTOP_SOURCE_STATE_PATHS
  ])
  if (diff == null || untrackedOutput == null) return null

  const untrackedEntries = untrackedOutput
    .split('\0')
    .filter(Boolean)
    .sort()
    .flatMap(relPath => {
      const absPath = path.join(repoRoot, relPath)
      try {
        const stat = fsImpl.statSync(absPath)
        if (!stat.isFile()) return []
        return [`${relPath}\t${stat.size}\t${hashFile(absPath, fsImpl)}`]
      } catch {
        return []
      }
    })

  return hashText(desktopSourceStateMaterial(diff, untrackedEntries))
}

module.exports = {
  DESKTOP_SOURCE_STATE_PATHS,
  EMPTY_DESKTOP_SOURCE_STATE_HASH,
  currentDesktopSourceStateHash,
  desktopSourceStateMaterial,
  hashText
}
