import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, renameSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { rebuildNodePty } from './rebuild-native.mjs'
import {
  assertNodePtyNativePayload,
  findNodePtyNativePayload,
  isHostTarget,
  nodeRuntimeArch,
  resolveNodePtyRoot,
  stageNodePty
} from './stage-native-deps.mjs'

function withTempRoot(run) {
  const root = mkdtempSync(join(tmpdir(), 'hermes-node-pty-'))
  try {
    run(root)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
}

function writePayload(dir, { spawnHelper = false } = {}) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'pty.node'), 'native binding')
  if (spawnHelper) writeFileSync(join(dir, 'spawn-helper'), 'native helper')
}

function writePackageFixture(root) {
  writeFileSync(join(root, 'package.json'), '{"main":"lib/index.js"}')
  mkdirSync(join(root, 'lib'), { recursive: true })
  writeFileSync(join(root, 'lib', 'index.js'), 'module.exports = {}')
}

test('nodeRuntimeArch normalizes electron-builder armv7l to Node arm', () => {
  assert.equal(nodeRuntimeArch('armv7l'), 'arm')
  assert.equal(nodeRuntimeArch('arm64'), 'arm64')
})

test('isHostTarget treats arm and armv7l as the same host architecture', () => {
  assert.equal(isHostTarget('linux', 'armv7l', { hostPlatform: 'linux', hostArch: 'arm' }), true)
  assert.equal(isHostTarget('linux', 'arm64', { hostPlatform: 'linux', hostArch: 'arm' }), false)
})

test('findNodePtyNativePayload accepts a target prebuild', () => {
  withTempRoot(root => {
    const payload = join(root, 'prebuilds', 'linux-x64')
    writePayload(payload)

    assert.equal(findNodePtyNativePayload({ root, platform: 'linux', arch: 'x64' }), payload)
  })
})

test('findNodePtyNativePayload accepts a local build output', () => {
  withTempRoot(root => {
    const payload = join(root, 'build', 'Release')
    writePayload(payload, { spawnHelper: process.platform === 'darwin' })

    assert.equal(findNodePtyNativePayload({ root, platform: process.platform, arch: process.arch }), payload)
  })
})

test('findNodePtyNativePayload matches node-pty runtime precedence', () => {
  withTempRoot(root => {
    const build = join(root, 'build', 'Release')
    const prebuild = join(root, 'prebuilds', 'linux-x64')
    writePayload(build)
    writePayload(prebuild)

    assert.equal(
      findNodePtyNativePayload({
        root,
        platform: 'linux',
        arch: 'x64',
        includeBuild: true
      }),
      build
    )
  })
})

test('findNodePtyNativePayload skips an incomplete Darwin build', () => {
  withTempRoot(root => {
    const build = join(root, 'build', 'Release')
    const prebuild = join(root, 'prebuilds', 'darwin-x64')
    writePayload(build)
    writePayload(prebuild, { spawnHelper: true })

    assert.equal(
      findNodePtyNativePayload({
        root,
        platform: 'darwin',
        arch: 'x64',
        includeBuild: true
      }),
      prebuild
    )
  })
})

test('findNodePtyNativePayload does not reuse a host build for a cross-target package', () => {
  withTempRoot(root => {
    writePayload(join(root, 'build', 'Release'), { spawnHelper: true })
    const targetPlatform = process.platform === 'linux' ? 'win32' : 'linux'

    assert.equal(findNodePtyNativePayload({ root, platform: targetPlatform, arch: process.arch }), undefined)
  })
})

test('assertNodePtyNativePayload fails when pty.node is absent', () => {
  withTempRoot(root => {
    assert.throws(
      () => assertNodePtyNativePayload(root, { platform: 'linux', arch: 'x64' }),
      /no usable native payload/
    )
  })
})

test('findNodePtyNativePayload rejects an empty pty.node', () => {
  withTempRoot(root => {
    const payload = join(root, 'prebuilds', 'linux-x64')
    mkdirSync(payload, { recursive: true })
    writeFileSync(join(payload, 'pty.node'), '')

    assert.equal(findNodePtyNativePayload({ root, platform: 'linux', arch: 'x64' }), undefined)
  })
})

test('findNodePtyNativePayload rejects an empty Darwin spawn-helper', () => {
  withTempRoot(root => {
    const payload = join(root, 'prebuilds', 'darwin-x64')
    writePayload(payload)
    writeFileSync(join(payload, 'spawn-helper'), '')

    assert.equal(findNodePtyNativePayload({ root, platform: 'darwin', arch: 'x64' }), undefined)
  })
})

test('assertNodePtyNativePayload accepts a staged build payload', () => {
  withTempRoot(root => {
    const payload = join(root, 'build', 'Release')
    writePayload(payload, { spawnHelper: process.platform === 'darwin' })

    assert.equal(assertNodePtyNativePayload(root, { platform: process.platform, arch: process.arch }), payload)
  })
})

test('stageNodePty copies only the payload selected at runtime', () => {
  withTempRoot(root => {
    const srcRoot = join(root, 'source')
    const destRoot = join(root, 'staged')
    const build = join(srcRoot, 'build', 'Release')
    const prebuild = join(srcRoot, 'prebuilds', `${process.platform}-${nodeRuntimeArch(process.arch)}`)
    mkdirSync(srcRoot, { recursive: true })
    writePackageFixture(srcRoot)
    writePayload(build, { spawnHelper: process.platform === 'darwin' })
    writePayload(prebuild, { spawnHelper: process.platform === 'darwin' })

    stageNodePty({ srcRoot, destRoot })

    assert.equal(existsSync(join(destRoot, 'build', 'Release', 'pty.node')), true)
    assert.equal(existsSync(join(destRoot, 'prebuilds')), false)
  })
})

test('stageNodePty omits an incomplete build when a valid prebuild is selected', () => {
  withTempRoot(root => {
    const srcRoot = join(root, 'source')
    const destRoot = join(root, 'staged')
    const build = join(srcRoot, 'build', 'Release')
    const runtimeArch = nodeRuntimeArch(process.arch)
    const prebuild = join(srcRoot, 'prebuilds', `${process.platform}-${runtimeArch}`)
    mkdirSync(build, { recursive: true })
    writePackageFixture(srcRoot)
    writePayload(prebuild, { spawnHelper: process.platform === 'darwin' })

    stageNodePty({ srcRoot, destRoot })

    assert.equal(existsSync(join(destRoot, 'build')), false)
    assert.equal(existsSync(join(destRoot, 'prebuilds', `${process.platform}-${runtimeArch}`, 'pty.node')), true)
  })
})

test(
  'rebuildNodePty recreates a missing Linux payload through the npm workspace',
  { skip: process.platform !== 'linux' },
  async t => {
    const root = resolveNodePtyRoot()
    const prebuild = join(root, 'prebuilds', `linux-${nodeRuntimeArch(process.arch)}`, 'pty.node')
    if (existsSync(prebuild)) {
      t.skip('the installed node-pty version already provides a Linux prebuild')
      return
    }

    const buildDir = join(root, 'build')
    const backupDir = join(root, `.hermes-build-backup-${process.pid}`)
    const hadBuild = existsSync(buildDir)
    rmSync(backupDir, { recursive: true, force: true })
    if (hadBuild) renameSync(buildDir, backupDir)

    try {
      assert.equal(findNodePtyNativePayload(), undefined)
      assert.equal(await rebuildNodePty(), true)
      assert.equal(existsSync(join(buildDir, 'Release', 'pty.node')), true)
    } finally {
      rmSync(buildDir, { recursive: true, force: true })
      if (hadBuild) renameSync(backupDir, buildDir)
    }
  }
)
