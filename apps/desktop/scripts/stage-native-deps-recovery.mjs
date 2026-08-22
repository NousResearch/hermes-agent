#!/usr/bin/env node
// Isolated dependency-realization wrapper for stage-native-deps.mjs.
//
// The normal Desktop build stages get-windows from the repository's installed
// dependency tree. On Windows, an interrupted/in-place npm extraction can leave
// node_modules/get-windows present but unresolvable (for example, package.json
// is missing). Reusing that tree makes every later Desktop rebuild fail at the
// same stage. This wrapper preserves the fail-closed native staging contract,
// but realizes the exact package in a fresh temporary npm prefix and stages
// from that verified root instead of mutating or trusting active node_modules.
//
// Recovery is intentionally two-phase:
//   1. install with lifecycle scripts disabled;
//   2. prove the realized dependency graph is a subset of the committed lock
//      closure and carries the repository override policy;
//   3. only then run get-windows' lifecycle script.
//
// No unreviewed registry/transitive graph may execute code during recovery.

import { spawnSync } from 'node:child_process'
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { stageGetWindows, stageNodePty } from './stage-native-deps.mjs'
import { isMain } from './utils.mjs'

const require = createRequire(import.meta.url)
const SCRIPT_DIR = dirname(fileURLToPath(import.meta.url))
const DEFAULT_REPOSITORY_ROOT = resolve(SCRIPT_DIR, '..', '..', '..')

export const GET_WINDOWS_RECOVERY_VERSION = '9.3.0'
export const GET_WINDOWS_MISSING_ROOT_MARKER =
  '[stage-native-deps] get-windows is not installed; cannot stage its '

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error)
}

export function isMissingGetWindowsPackageError(error) {
  return errorMessage(error).includes(GET_WINDOWS_MISSING_ROOT_MARKER)
}

export function canRecoverGetWindowsPackage({
  platform,
  arch,
  hostPlatform = process.platform,
  hostArch = process.arch
}) {
  if (platform !== hostPlatform) {
    return false
  }

  if (platform === 'darwin') {
    // get-windows' macOS helper is universal.
    return true
  }

  if (platform === 'win32') {
    // The package's node-pre-gyp lifecycle realizes a host-architecture
    // binding. Only consume it when the requested package architecture is the
    // same one the active Node process can actually verify and execute.
    return arch === hostArch && (arch === 'x64' || arch === 'ia32')
  }

  return false
}

function cleanupRecoveryRoot(recoveryRoot) {
  try {
    rmSync(recoveryRoot, {
      force: true,
      maxRetries: 5,
      recursive: true,
      retryDelay: 100
    })
    return true
  } catch (error) {
    console.warn(
      `[stage-native-deps] could not remove isolated get-windows recovery root ${recoveryRoot}: ${errorMessage(error)}`
    )
    return false
  }
}

function lockSignature(entry) {
  if (
    !entry ||
    typeof entry.version !== 'string' ||
    typeof entry.resolved !== 'string' ||
    typeof entry.integrity !== 'string'
  ) {
    return undefined
  }
  return `${entry.version}\u0000${entry.resolved}\u0000${entry.integrity}`
}

function resolveLockedDependency(packages, fromPath, dependencyName) {
  let prefix = fromPath
  while (true) {
    const candidate = prefix
      ? `${prefix}/node_modules/${dependencyName}`
      : `node_modules/${dependencyName}`
    if (packages[candidate]) {
      return candidate
    }
    if (!prefix) {
      return undefined
    }

    const nestedMarker = prefix.lastIndexOf('/node_modules/')
    if (nestedMarker >= 0) {
      prefix = prefix.slice(0, nestedMarker)
    } else if (prefix.startsWith('node_modules/')) {
      prefix = ''
    } else {
      return undefined
    }
  }
}

export function committedGetWindowsClosure(repositoryLock) {
  const packages = repositoryLock && repositoryLock.packages
  if (!packages || typeof packages !== 'object') {
    throw new Error('[stage-native-deps] repository package-lock.json has no packages graph')
  }

  const rootPath = 'node_modules/get-windows'
  const rootEntry = packages[rootPath]
  if (!rootEntry || rootEntry.version !== GET_WINDOWS_RECOVERY_VERSION) {
    throw new Error(
      `[stage-native-deps] repository lock does not authorize get-windows@${GET_WINDOWS_RECOVERY_VERSION}`
    )
  }
  if (!lockSignature(rootEntry)) {
    throw new Error(
      '[stage-native-deps] repository get-windows lock entry lacks resolved/integrity provenance'
    )
  }

  const visited = new Set()
  const queue = [rootPath]
  while (queue.length > 0) {
    const packagePath = queue.shift()
    if (!packagePath || visited.has(packagePath)) {
      continue
    }
    const entry = packages[packagePath]
    if (!entry) {
      throw new Error(
        `[stage-native-deps] repository lock closure references missing package ${packagePath}`
      )
    }
    visited.add(packagePath)

    const dependencySets = [
      entry.dependencies,
      entry.optionalDependencies,
      entry.peerDependencies
    ]
    for (const dependencySet of dependencySets) {
      if (!dependencySet || typeof dependencySet !== 'object') {
        continue
      }
      for (const dependencyName of Object.keys(dependencySet)) {
        const resolvedPath = resolveLockedDependency(packages, packagePath, dependencyName)
        if (resolvedPath && !visited.has(resolvedPath)) {
          queue.push(resolvedPath)
        }
      }
    }
  }

  return visited
}

function overrideVersion(overrides, packageName) {
  const value = overrides && overrides[packageName]
  if (typeof value === 'string') {
    return value
  }
  if (value && typeof value === 'object' && typeof value['.'] === 'string') {
    return value['.']
  }
  return undefined
}

export function verifyRecoveryGraphAgainstRepository({
  recoveryLock,
  repositoryLock,
  repositoryOverrides
}) {
  const recoveryPackages = recoveryLock && recoveryLock.packages
  const repositoryPackages = repositoryLock && repositoryLock.packages
  if (!recoveryPackages || typeof recoveryPackages !== 'object') {
    throw new Error('[stage-native-deps] isolated recovery did not produce a package-lock graph')
  }
  if (!repositoryPackages || typeof repositoryPackages !== 'object') {
    throw new Error('[stage-native-deps] repository package-lock graph is unavailable')
  }

  const closurePaths = committedGetWindowsClosure(repositoryLock)
  const allowedSignatures = new Set()
  for (const packagePath of closurePaths) {
    const signature = lockSignature(repositoryPackages[packagePath])
    if (signature) {
      allowedSignatures.add(signature)
    }
  }

  const recoveryRootEntry = recoveryPackages['node_modules/get-windows']
  const repositoryRootEntry = repositoryPackages['node_modules/get-windows']
  if (
    lockSignature(recoveryRootEntry) !== lockSignature(repositoryRootEntry) ||
    recoveryRootEntry?.version !== GET_WINDOWS_RECOVERY_VERSION
  ) {
    throw new Error(
      '[stage-native-deps] isolated get-windows artifact does not match committed resolved/integrity provenance'
    )
  }

  for (const [packagePath, entry] of Object.entries(recoveryPackages)) {
    if (!packagePath) {
      continue
    }
    const signature = lockSignature(entry)
    if (!signature || !allowedSignatures.has(signature)) {
      throw new Error(
        `[stage-native-deps] isolated recovery package ${packagePath} is not present in the committed get-windows dependency closure`
      )
    }
  }

  const tarOverride = overrideVersion(repositoryOverrides, 'tar')
  if (tarOverride) {
    for (const [packagePath, entry] of Object.entries(recoveryPackages)) {
      if (
        packagePath === 'node_modules/tar' ||
        packagePath.endsWith('/node_modules/tar')
      ) {
        if (entry?.version !== tarOverride) {
          throw new Error(
            `[stage-native-deps] isolated recovery violates repository tar override: ${entry?.version || 'missing'} != ${tarOverride}`
          )
        }
      }
    }
  }

  return true
}

function readRepositoryDependencyAuthority(repositoryRoot) {
  const repositoryManifest = JSON.parse(
    readFileSync(join(repositoryRoot, 'package.json'), 'utf8')
  )
  const repositoryLock = JSON.parse(
    readFileSync(join(repositoryRoot, 'package-lock.json'), 'utf8')
  )
  return {
    overrides:
      repositoryManifest.overrides && typeof repositoryManifest.overrides === 'object'
        ? repositoryManifest.overrides
        : {},
    repositoryLock
  }
}

export function recoverGetWindowsPackage({
  platform = process.platform,
  arch = process.arch,
  npmExecPath = process.env.npm_execpath,
  run = spawnSync,
  tempParent = tmpdir(),
  repositoryRoot = DEFAULT_REPOSITORY_ROOT
} = {}) {
  if (!npmExecPath) {
    throw new Error(
      '[stage-native-deps] cannot recover get-windows: npm_execpath is unavailable; run the Desktop build through npm'
    )
  }

  const { overrides, repositoryLock } = readRepositoryDependencyAuthority(repositoryRoot)
  // Fail before network/process work if the committed lock itself cannot prove
  // the recovery root.
  committedGetWindowsClosure(repositoryLock)

  const recoveryRoot = mkdtempSync(join(tempParent, 'hermes-get-windows-'))
  let completed = false

  try {
    const recoveryManifest = {
      name: 'hermes-get-windows-recovery',
      private: true,
      version: '0.0.0',
      dependencies: {
        'get-windows': GET_WINDOWS_RECOVERY_VERSION
      },
      overrides,
      allowScripts: {
        [`get-windows@${GET_WINDOWS_RECOVERY_VERSION}`]: true
      }
    }
    writeFileSync(
      join(recoveryRoot, 'package.json'),
      `${JSON.stringify(recoveryManifest, null, 2)}\n`,
      'utf8'
    )

    const npmEnv = {
      ...process.env,
      npm_config_arch: arch,
      npm_config_platform: platform,
      npm_config_target_arch: arch
    }

    // Phase 1: materialize bytes only. No dependency lifecycle script is
    // allowed to execute before the realized graph is checked against the
    // repository's committed lock and overrides.
    const installResult = run(
      process.execPath,
      [
        npmExecPath,
        'install',
        '--workspaces=false',
        '--include=optional',
        '--ignore-scripts=true',
        '--no-audit',
        '--no-fund',
        '--package-lock=true',
        '--prefer-online'
      ],
      {
        cwd: recoveryRoot,
        env: npmEnv,
        stdio: 'inherit'
      }
    )

    if (installResult.error) {
      throw new Error(
        `[stage-native-deps] isolated get-windows recovery could not start npm: ${installResult.error.message}`
      )
    }
    if (installResult.status !== 0) {
      throw new Error(
        `[stage-native-deps] isolated get-windows recovery install exited with ${installResult.status}`
      )
    }

    const recoveryLockPath = join(recoveryRoot, 'package-lock.json')
    const recoveryLock = JSON.parse(readFileSync(recoveryLockPath, 'utf8'))
    verifyRecoveryGraphAgainstRepository({
      recoveryLock,
      repositoryLock,
      repositoryOverrides: overrides
    })

    // Phase 2: now that every installed registry artifact is authorized by the
    // committed get-windows closure, allow only get-windows' lifecycle to run.
    const rebuildResult = run(
      process.execPath,
      [
        npmExecPath,
        'rebuild',
        'get-windows',
        '--workspaces=false',
        '--ignore-scripts=false',
        '--no-audit',
        '--no-fund'
      ],
      {
        cwd: recoveryRoot,
        env: npmEnv,
        stdio: 'inherit'
      }
    )
    if (rebuildResult.error) {
      throw new Error(
        `[stage-native-deps] isolated get-windows lifecycle could not start npm: ${rebuildResult.error.message}`
      )
    }
    if (rebuildResult.status !== 0) {
      throw new Error(
        `[stage-native-deps] isolated get-windows lifecycle exited with ${rebuildResult.status}`
      )
    }

    // Lifecycle execution must not rewrite the dependency authority we just
    // attested.
    verifyRecoveryGraphAgainstRepository({
      recoveryLock: JSON.parse(readFileSync(recoveryLockPath, 'utf8')),
      repositoryLock,
      repositoryOverrides: overrides
    })

    let packageRoot
    try {
      // get-windows does not export package.json; resolve its root entry and
      // validate the manifest beside it.
      packageRoot = dirname(
        require.resolve('get-windows', {
          paths: [recoveryRoot]
        })
      )
    } catch (error) {
      throw new Error(
        `[stage-native-deps] isolated get-windows recovery completed without an importable package: ${errorMessage(error)}`
      )
    }

    const installedVersion = JSON.parse(
      readFileSync(join(packageRoot, 'package.json'), 'utf8')
    ).version
    if (installedVersion !== GET_WINDOWS_RECOVERY_VERSION) {
      throw new Error(
        `[stage-native-deps] isolated get-windows recovery resolved ${installedVersion}; expected ${GET_WINDOWS_RECOVERY_VERSION}`
      )
    }

    completed = true
    let cleaned = false
    return {
      packageRoot,
      recoveryRoot,
      cleanup() {
        if (cleaned) {
          return true
        }
        cleaned = cleanupRecoveryRoot(recoveryRoot)
        return cleaned
      }
    }
  } finally {
    if (!completed) {
      cleanupRecoveryRoot(recoveryRoot)
    }
  }
}

export function stageGetWindowsWithRecovery({
  platform = process.platform,
  arch = process.arch,
  hostPlatform = process.platform,
  hostArch = process.arch,
  recover = recoverGetWindowsPackage,
  stage = stageGetWindows
} = {}) {
  try {
    return stage({ platform, arch })
  } catch (error) {
    if (
      !isMissingGetWindowsPackageError(error) ||
      !canRecoverGetWindowsPackage({ platform, arch, hostPlatform, hostArch })
    ) {
      throw error
    }

    console.warn(
      `[stage-native-deps] get-windows package root is missing or corrupt for ${platform}-${arch}; ` +
        `recovering exact ${GET_WINDOWS_RECOVERY_VERSION} in an isolated npm prefix`
    )
    const recovery = recover({ platform, arch })

    try {
      return stage({
        platform,
        arch,
        resolveRoot: () => recovery.packageRoot
      })
    } finally {
      recovery.cleanup()
    }
  }
}

export function stageNativeDeps({ platform = process.platform, arch = process.arch } = {}) {
  const nodePty = stageNodePty({ platform, arch })
  const getWindows = stageGetWindowsWithRecovery({ platform, arch })
  return { getWindows, nodePty }
}

if (isMain(import.meta.url)) {
  const [platform, arch] = process.argv.slice(2)
  stageNativeDeps({ platform, arch })
}
