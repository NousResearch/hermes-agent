import { existsSync, realpathSync } from 'node:fs'
import path from 'node:path'

export const BUILDER_REEXEC_GUARD_ENV = 'HERMES_ELECTRON_BUILDER_REEXEC'
export const MIN_BUILDER_NODE_VERSION = '22.22.0'

function normalizeExecutablePath(filePath, platform, realpath) {
  if (!filePath || typeof filePath !== 'string') {
    return undefined
  }
  let normalized
  try {
    normalized = realpath(filePath)
  } catch {
    normalized = path.resolve(filePath)
  }
  normalized = normalized.replace(/[\\/]+$/, '')
  return platform === 'win32' ? normalized.toLowerCase() : normalized
}

/**
 * npm records the exact Node executable that launched it in npm_node_execpath.
 * On Windows, package scripts launched through cmd.exe can resolve a different
 * bare `node` from PATH (fnm/system Node), even while npm itself is running on
 * Hermes-managed Node. Select npm's runtime for one bounded self-reexec so
 * install and package generations cannot silently use different interpreters.
 */
export function selectNpmNodeRuntime({
  currentExecPath,
  npmNodeExecPath,
  guardValue,
  platform = process.platform,
  exists = existsSync,
  realpath = realpathSync.native
}) {
  if (platform !== 'win32' || guardValue === '1') {
    return undefined
  }
  if (!npmNodeExecPath || !exists(npmNodeExecPath)) {
    return undefined
  }
  const current = normalizeExecutablePath(currentExecPath, platform, realpath)
  const npmSelected = normalizeExecutablePath(npmNodeExecPath, platform, realpath)
  if (!current || !npmSelected || current === npmSelected) {
    return undefined
  }
  return npmNodeExecPath
}

export function nodeVersionAtLeast(version, minimum = MIN_BUILDER_NODE_VERSION) {
  const parse = value => String(value || '').split('.').map(part => Number.parseInt(part, 10))
  const actual = parse(version)
  const required = parse(minimum)
  if (actual.length < 2 || actual.some(part => !Number.isFinite(part))) {
    return false
  }
  for (let index = 0; index < Math.max(actual.length, required.length); index += 1) {
    const left = actual[index] || 0
    const right = required[index] || 0
    if (left !== right) {
      return left > right
    }
  }
  return true
}

export function desktopBuilderRuntimeProblem({
  version,
  execPath,
  requireModuleSupported,
  minimum = MIN_BUILDER_NODE_VERSION
}) {
  if (!nodeVersionAtLeast(version, minimum)) {
    return `Node ${version || 'unknown'} at ${execPath || 'unknown path'} is too old; Hermes Desktop packaging requires Node >=${minimum}`
  }
  if (requireModuleSupported !== true) {
    return `Node ${version} at ${execPath || 'unknown path'} cannot require ESM modules; remove --no-experimental-require-module and retry`
  }
  return undefined
}
