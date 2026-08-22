import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'node:test'

import {
  GET_WINDOWS_MISSING_ROOT_MARKER,
  GET_WINDOWS_RECOVERY_VERSION,
  canRecoverGetWindowsPackage,
  committedGetWindowsClosure,
  recoverGetWindowsPackage,
  stageGetWindowsWithRecovery,
  verifyRecoveryGraphAgainstRepository
} from './stage-native-deps-recovery.mjs'

function tempParent() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-get-windows-recovery-test-'))
}

function packageEntry(version, name) {
  return {
    version,
    resolved: `https://registry.npmjs.org/${name}/-/${name}-${version}.tgz`,
    integrity: `sha512-${name}-${version}`
  }
}

function writeRepositoryAuthority(root) {
  const overrides = { tar: '7.5.22' }
  const getWindows = {
    ...packageEntry(GET_WINDOWS_RECOVERY_VERSION, 'get-windows'),
    optionalDependencies: {
      'node-gyp': '^10.2.0'
    }
  }
  const nodeGyp = {
    ...packageEntry('10.3.1', 'node-gyp'),
    dependencies: {
      tar: '^6.2.1'
    }
  }
  const tar = packageEntry('7.5.22', 'tar')
  const repositoryLock = {
    lockfileVersion: 3,
    packages: {
      '': {},
      'node_modules/get-windows': getWindows,
      'node_modules/get-windows/node_modules/node-gyp': nodeGyp,
      'node_modules/tar': tar
    }
  }

  fs.mkdirSync(root, { recursive: true })
  fs.writeFileSync(
    path.join(root, 'package.json'),
    `${JSON.stringify({ name: 'fixture', private: true, overrides }, null, 2)}\n`
  )
  fs.writeFileSync(
    path.join(root, 'package-lock.json'),
    `${JSON.stringify(repositoryLock, null, 2)}\n`
  )
  return { overrides, repositoryLock }
}

function recoveryLockFrom(repositoryLock, mutate = undefined) {
  const packages = structuredClone(repositoryLock.packages)
  packages[''] = {
    name: 'hermes-get-windows-recovery',
    version: '0.0.0',
    dependencies: { 'get-windows': GET_WINDOWS_RECOVERY_VERSION }
  }
  if (mutate) {
    mutate(packages)
  }
  return { lockfileVersion: 3, packages }
}

function materializeRecoveredPackage(cwd) {
  const packageRoot = path.join(cwd, 'node_modules', 'get-windows')
  fs.mkdirSync(path.join(packageRoot, 'lib'), { recursive: true })
  fs.writeFileSync(
    path.join(packageRoot, 'package.json'),
    JSON.stringify({ name: 'get-windows', version: GET_WINDOWS_RECOVERY_VERSION })
  )
  fs.writeFileSync(path.join(packageRoot, 'index.js'), 'module.exports = {}\n')
}

test('committed closure includes only dependency identities reachable from get-windows', () => {
  const parent = tempParent()
  try {
    const { repositoryLock } = writeRepositoryAuthority(path.join(parent, 'repo'))
    repositoryLock.packages['node_modules/unrelated'] = packageEntry('1.0.0', 'unrelated')
    assert.deepEqual(
      [...committedGetWindowsClosure(repositoryLock)].sort(),
      [
        'node_modules/get-windows',
        'node_modules/get-windows/node_modules/node-gyp',
        'node_modules/tar'
      ]
    )
  } finally {
    fs.rmSync(parent, { force: true, recursive: true })
  }
})

test('isolated recovery materializes with scripts disabled, proves the lock graph, then rebuilds get-windows', () => {
  const parent = tempParent()
  const repositoryRoot = path.join(parent, 'repo')
  try {
    const { overrides, repositoryLock } = writeRepositoryAuthority(repositoryRoot)
    const invocations = []

    const recovery = recoverGetWindowsPackage({
      arch: 'x64',
      npmExecPath: '/fake/npm-cli.js',
      platform: 'win32',
      repositoryRoot,
      tempParent: parent,
      run(command, args, options) {
        invocations.push({ args: [...args], command, options })
        if (args[1] === 'install') {
          materializeRecoveredPackage(options.cwd)
          fs.writeFileSync(
            path.join(options.cwd, 'package-lock.json'),
            `${JSON.stringify(recoveryLockFrom(repositoryLock), null, 2)}\n`
          )
          return { status: 0 }
        }
        if (args[1] === 'rebuild') {
          return { status: 0 }
        }
        throw new Error(`unexpected npm action: ${args[1]}`)
      }
    })

    assert.equal(invocations.length, 2)
    const install = invocations[0]
    const rebuild = invocations[1]

    assert.equal(install.command, process.execPath)
    assert.equal(install.args[0], '/fake/npm-cli.js')
    assert.equal(install.args[1], 'install')
    assert.ok(install.args.includes('--workspaces=false'))
    assert.ok(install.args.includes('--include=optional'))
    assert.ok(install.args.includes('--ignore-scripts=true'))
    assert.ok(install.args.includes('--package-lock=true'))
    assert.equal(install.args.includes('--ignore-scripts=false'), false)

    assert.equal(rebuild.args[1], 'rebuild')
    assert.equal(rebuild.args[2], 'get-windows')
    assert.ok(rebuild.args.includes('--ignore-scripts=false'))

    assert.equal(install.options.cwd, recovery.recoveryRoot)
    assert.equal(install.options.env.npm_config_platform, 'win32')
    assert.equal(install.options.env.npm_config_arch, 'x64')

    const manifest = JSON.parse(
      fs.readFileSync(path.join(recovery.recoveryRoot, 'package.json'), 'utf8')
    )
    assert.deepEqual(manifest.dependencies, {
      'get-windows': GET_WINDOWS_RECOVERY_VERSION
    })
    assert.deepEqual(manifest.overrides, overrides)
    assert.deepEqual(manifest.allowScripts, {
      [`get-windows@${GET_WINDOWS_RECOVERY_VERSION}`]: true
    })
    assert.equal(
      recovery.packageRoot,
      path.join(recovery.recoveryRoot, 'node_modules', 'get-windows')
    )
    assert.ok(fs.existsSync(recovery.recoveryRoot))

    assert.equal(recovery.cleanup(), true)
    assert.equal(recovery.cleanup(), true)
    assert.equal(fs.existsSync(recovery.recoveryRoot), false)
  } finally {
    fs.rmSync(parent, { force: true, recursive: true })
  }
})

test('drifted transitive resolution is rejected before any lifecycle script executes', () => {
  const parent = tempParent()
  const repositoryRoot = path.join(parent, 'repo')
  try {
    const { repositoryLock } = writeRepositoryAuthority(repositoryRoot)
    const actions = []

    assert.throws(
      () =>
        recoverGetWindowsPackage({
          npmExecPath: '/fake/npm-cli.js',
          repositoryRoot,
          tempParent: parent,
          run(_command, args, options) {
            actions.push(args[1])
            if (args[1] !== 'install') {
              throw new Error('lifecycle must not execute after provenance failure')
            }
            materializeRecoveredPackage(options.cwd)
            const drifted = recoveryLockFrom(repositoryLock, packages => {
              packages['node_modules/get-windows/node_modules/node-gyp'] = {
                ...packages['node_modules/get-windows/node_modules/node-gyp'],
                resolved: 'https://registry.example.invalid/node-gyp-10.3.1.tgz',
                integrity: 'sha512-drifted'
              }
            })
            fs.writeFileSync(
              path.join(options.cwd, 'package-lock.json'),
              `${JSON.stringify(drifted, null, 2)}\n`
            )
            return { status: 0 }
          }
        }),
      /not present in the committed get-windows dependency closure/
    )
    assert.deepEqual(actions, ['install'])
    assert.deepEqual(
      fs.readdirSync(parent).filter(name => name.startsWith('hermes-get-windows-')),
      []
    )
  } finally {
    fs.rmSync(parent, { force: true, recursive: true })
  }
})

test('repository tar override is part of recovery authority and cannot drift', () => {
  const parent = tempParent()
  try {
    const { overrides, repositoryLock } = writeRepositoryAuthority(path.join(parent, 'repo'))
    const drifted = recoveryLockFrom(repositoryLock, packages => {
      packages['node_modules/tar'] = packageEntry('6.2.1', 'tar')
    })

    assert.throws(
      () =>
        verifyRecoveryGraphAgainstRepository({
          recoveryLock: drifted,
          repositoryLock,
          repositoryOverrides: overrides
        }),
      /tar override|committed get-windows dependency closure/
    )

    assert.equal(
      verifyRecoveryGraphAgainstRepository({
        recoveryLock: recoveryLockFrom(repositoryLock),
        repositoryLock,
        repositoryOverrides: overrides
      }),
      true
    )
  } finally {
    fs.rmSync(parent, { force: true, recursive: true })
  }
})

test('failed isolated install is removed and cannot poison a later update', () => {
  const parent = tempParent()
  const repositoryRoot = path.join(parent, 'repo')
  try {
    writeRepositoryAuthority(repositoryRoot)
    assert.throws(
      () =>
        recoverGetWindowsPackage({
          npmExecPath: '/fake/npm-cli.js',
          repositoryRoot,
          run: () => ({ status: 1 }),
          tempParent: parent
        }),
      /isolated get-windows recovery install exited with 1/
    )
    assert.deepEqual(
      fs.readdirSync(parent).filter(name => name.startsWith('hermes-get-windows-')),
      []
    )
  } finally {
    fs.rmSync(parent, { force: true, recursive: true })
  }
})

test('supported native Windows staging retries with the recovered package root', () => {
  const calls = []
  let cleaned = false

  const result = stageGetWindowsWithRecovery({
    arch: 'x64',
    hostArch: 'x64',
    hostPlatform: 'win32',
    platform: 'win32',
    recover: () => ({
      cleanup() {
        cleaned = true
      },
      packageRoot: 'C:\\Temp\\isolated\\node_modules\\get-windows'
    }),
    stage(options) {
      calls.push(options)
      if (calls.length === 1) {
        throw new Error(`${GET_WINDOWS_MISSING_ROOT_MARKER}win32-x64 native payload`)
      }
      assert.equal(
        options.resolveRoot(),
        'C:\\Temp\\isolated\\node_modules\\get-windows'
      )
      return 'staged'
    }
  })

  assert.equal(result, 'staged')
  assert.equal(calls.length, 2)
  assert.equal(cleaned, true)
})

test('recovered temporary dependency is removed when the second staging attempt fails', () => {
  let calls = 0
  let cleaned = false

  assert.throws(
    () =>
      stageGetWindowsWithRecovery({
        arch: 'x64',
        hostArch: 'x64',
        hostPlatform: 'win32',
        platform: 'win32',
        recover: () => ({
          cleanup() {
            cleaned = true
          },
          packageRoot: 'C:\\Temp\\isolated\\node_modules\\get-windows'
        }),
        stage() {
          calls += 1
          if (calls === 1) {
            throw new Error(`${GET_WINDOWS_MISSING_ROOT_MARKER}win32-x64 native payload`)
          }
          throw new Error('recovered package has no binding')
        }
      }),
    /recovered package has no binding/
  )
  assert.equal(cleaned, true)
})

test('unrelated staging failures are never converted into dependency recovery', () => {
  let recovered = false

  assert.throws(
    () =>
      stageGetWindowsWithRecovery({
        recover() {
          recovered = true
          throw new Error('should not run')
        },
        stage() {
          throw new Error('native binary platform mismatch')
        }
      }),
    /native binary platform mismatch/
  )
  assert.equal(recovered, false)
})

test('cross-platform and unsupported Windows architecture requests remain fail-closed', () => {
  assert.equal(
    canRecoverGetWindowsPackage({
      arch: 'x64',
      hostArch: 'x64',
      hostPlatform: 'linux',
      platform: 'win32'
    }),
    false
  )
  assert.equal(
    canRecoverGetWindowsPackage({
      arch: 'arm64',
      hostArch: 'arm64',
      hostPlatform: 'win32',
      platform: 'win32'
    }),
    false
  )
  assert.equal(
    canRecoverGetWindowsPackage({
      arch: 'x64',
      hostArch: 'x64',
      hostPlatform: 'win32',
      platform: 'win32'
    }),
    true
  )
})

test('Desktop build and electron-builder hook both consume the recovery owner', () => {
  const manifest = JSON.parse(
    fs.readFileSync(new URL('../package.json', import.meta.url), 'utf8')
  )

  assert.match(manifest.scripts.build, /stage-native-deps-recovery\.mjs$/)
  assert.match(
    manifest.scripts['check:test:desktop:all'],
    /stage-native-deps-recovery\.test\.mjs/
  )
  assert.equal(manifest.build.beforePack, 'scripts/before-pack-recovery.mjs')
})
