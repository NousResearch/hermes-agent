'use strict'

/**
 * Build the Rust install manager for desktop release packaging.
 */

const { spawnSync } = require('node:child_process')
const path = require('node:path')

const APP_ROOT = path.resolve(__dirname, '..')
const REPO_ROOT = path.resolve(APP_ROOT, '..', '..')

function normalizeTargetPlatform(platform) {
  if (platform === 'win') return 'win32'
  if (platform === 'mac') return 'darwin'
  return platform
}

function buildCargoArgs(repoRoot = REPO_ROOT) {
  return [
    'build',
    '--release',
    '--manifest-path',
    path.join(repoRoot, 'apps', 'hermes-manager', 'Cargo.toml')
  ]
}

function shouldAllowHostBuild({ hostPlatform = process.platform, targetPlatform } = {}) {
  const normalizedHost = normalizeTargetPlatform(hostPlatform)
  const normalizedTarget = normalizeTargetPlatform(targetPlatform || hostPlatform)
  return normalizedHost === normalizedTarget
}

function main() {
  const targetPlatform = normalizeTargetPlatform(process.env.HERMES_DESKTOP_TARGET_PLATFORM || process.platform)
  if (!shouldAllowHostBuild({ targetPlatform })) {
    throw new Error(
      `Cannot build hermes-manager for ${targetPlatform} from host ${process.platform}. ` +
        'Build the desktop release on the target OS or add an explicit Rust cross target.'
    )
  }

  const result = spawnSync('cargo', buildCargoArgs(), {
    cwd: REPO_ROOT,
    stdio: 'inherit',
    windowsHide: true
  })
  if (result.error) throw result.error
  if (result.status !== 0) {
    throw new Error(`cargo build failed with exit code ${result.status}`)
  }
}

module.exports = { buildCargoArgs, normalizeTargetPlatform, shouldAllowHostBuild }

if (require.main === module) {
  try {
    main()
  } catch (error) {
    console.error(`[build-hermes-manager] ${error.message}`)
    process.exit(1)
  }
}

