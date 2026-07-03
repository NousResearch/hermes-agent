'use strict'

/**
 * Resolve mimalloc preload env vars for the Hermes dashboard Python child.
 *
 * The shared library is staged under build/native-deps/mimalloc/<platform>-<arch>/
 * by scripts/stage-native-deps.cjs (best-effort — skipped when not found locally).
 */

const fs = require('node:fs')
const path = require('node:path')

const MIMALLOC_ENV = Object.freeze({
  MIMALLOC_PURGE_DELAY: '0',
  MIMALLOC_ARENA_EAGER_COMMIT: '0'
})

function mimallocLibBasename(platform, arch) {
  if (platform === 'darwin') {
    return 'libmimalloc.dylib'
  }
  if (platform === 'win32') {
    return 'mimalloc-redirect.dll'
  }
  return 'libmimalloc.so'
}

function resolveMimallocLibPath({ resourcesPath, appRoot, platform, arch }) {
  const libName = mimallocLibBasename(platform, arch)
  const subdir = `${platform === 'darwin' ? 'darwin' : platform === 'win32' ? 'win32' : 'linux'}-${arch}`
  const candidates = []

  if (resourcesPath) {
    candidates.push(path.join(resourcesPath, 'native-deps', 'mimalloc', subdir, libName))
  }
  if (appRoot) {
    candidates.push(path.join(appRoot, 'build', 'native-deps', 'mimalloc', subdir, libName))
  }

  for (const candidate of candidates) {
    try {
      if (candidate && fs.existsSync(candidate)) {
        return candidate
      }
    } catch {
      // ignore
    }
  }
  return null
}

/**
 * @param {object} opts
 * @param {string} [opts.resourcesPath] process.resourcesPath (packaged)
 * @param {string} [opts.appRoot] apps/desktop root (dev)
 * @param {boolean} [opts.enabled] when false, returns {}
 */
function buildAllocatorEnv({
  resourcesPath,
  appRoot,
  enabled = true,
  platform = process.platform,
  arch = process.arch
} = {}) {
  if (!enabled || platform === 'win32') {
    return {}
  }

  const libPath = resolveMimallocLibPath({ resourcesPath, appRoot, platform, arch })
  if (!libPath) {
    return {}
  }

  const env = { ...MIMALLOC_ENV }
  if (platform === 'darwin') {
    env.DYLD_INSERT_LIBRARIES = libPath
  } else if (platform === 'linux') {
    env.LD_PRELOAD = libPath
  }
  return env
}

module.exports = {
  MIMALLOC_ENV,
  buildAllocatorEnv,
  mimallocLibBasename,
  resolveMimallocLibPath
}
