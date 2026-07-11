import assert from 'node:assert/strict'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import test from 'node:test'
import { rebuildNodePty } from './rebuild-native.mjs'
import {
  assertNodePtyNativePayload,
  copyDirRecursive,
  findNodePtyNativePayload,
  findNodePtyNativePayloads,
  isHostTarget,
  nativeBinaryMatchesTarget,
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

function nativeBinary(platform, arch) {
  const runtimeArch = nodeRuntimeArch(arch)
  if (platform === 'linux') {
    const data = Buffer.alloc(64)
    Buffer.from([0x7f, 0x45, 0x4c, 0x46]).copy(data)
    data[4] = ['x64', 'arm64'].includes(runtimeArch) ? 2 : 1
    data[5] = 1
    data.writeUInt16LE({ ia32: 3, x64: 62, arm: 40, arm64: 183 }[runtimeArch], 18)
    return data
  }
  if (platform === 'win32') {
    const data = Buffer.alloc(128)
    data.write('MZ', 0, 'ascii')
    data.writeUInt32LE(64, 0x3c)
    data.write('PE\0\0', 64, 'ascii')
    data.writeUInt16LE({ ia32: 0x14c, x64: 0x8664, arm: 0x1c4, arm64: 0xaa64 }[runtimeArch], 68)
    return data
  }
  if (platform === 'darwin') {
    const data = Buffer.alloc(32)
    data.writeUInt32BE(0xcffaedfe, 0)
    data.writeUInt32LE({ ia32: 7, x64: 0x01000007, arm: 12, arm64: 0x0100000c }[runtimeArch], 4)
    return data
  }
  throw new Error(`unsupported fixture target ${platform}-${arch}`)
}

function writePayload(
  dir,
  { platform = process.platform, arch = process.arch, spawnHelper = false, helperMode = 0o644 } = {}
) {
  mkdirSync(dir, { recursive: true })
  writeFileSync(join(dir, 'pty.node'), nativeBinary(platform, arch))
  if (spawnHelper) writeFileSync(join(dir, 'spawn-helper'), 'native helper', { mode: helperMode })
}

function writePackageFixture(root) {
  writeFileSync(join(root, 'package.json'), '{"main":"lib/index.js"}')
  mkdirSync(join(root, 'lib'), { recursive: true })
  writeFileSync(join(root, 'lib', 'index.js'), 'module.exports = {}')
}

function usesRecursiveCpSync(source) {
  const code = source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '')
  const call = 'cpSync('
  for (let start = code.indexOf(call); start !== -1; start = code.indexOf(call, start + 1)) {
    let depth = 0
    let index = start + call.length - 1
    for (; index < code.length; index++) {
      if (code[index] === '(') depth++
      else if (code[index] === ')' && --depth === 0) break
    }
    if (/\brecursive\b/.test(code.slice(start + call.length, index))) return true
  }
  return false
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
    writePayload(payload, { platform: 'linux', arch: 'x64' })

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
    writePayload(build, { platform: 'linux', arch: 'x64' })
    writePayload(prebuild, { platform: 'linux', arch: 'x64' })

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
    writePayload(build, { platform: 'darwin', arch: 'x64' })
    writePayload(prebuild, { platform: 'darwin', arch: 'x64', spawnHelper: true })

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
    writePayload(payload, { platform: 'darwin', arch: 'x64' })
    writeFileSync(join(payload, 'spawn-helper'), '')

    assert.equal(findNodePtyNativePayload({ root, platform: 'darwin', arch: 'x64' }), undefined)
  })
})

test('assertNodePtyNativePayload accepts a staged build payload', () => {
  withTempRoot(root => {
    const payload = join(root, 'build', 'Release')
    writePayload(payload, { spawnHelper: process.platform === 'darwin' })

    assert.deepEqual(assertNodePtyNativePayload(root, { platform: process.platform, arch: process.arch }), [payload])
  })
})

test('stageNodePty preserves every usable runtime fallback in loader order', () => {
  withTempRoot(root => {
    const srcRoot = join(root, 'source')
    const destRoot = join(root, 'staged')
    const build = join(srcRoot, 'build', 'Release')
    const runtimeArch = nodeRuntimeArch(process.arch)
    const prebuild = join(srcRoot, 'prebuilds', `${process.platform}-${runtimeArch}`)
    mkdirSync(srcRoot, { recursive: true })
    writePackageFixture(srcRoot)
    writePayload(build, { spawnHelper: process.platform === 'darwin' })
    writePayload(prebuild, { spawnHelper: process.platform === 'darwin' })

    stageNodePty({ srcRoot, destRoot })

    assert.equal(existsSync(join(destRoot, 'build', 'Release', 'pty.node')), true)
    assert.equal(existsSync(join(destRoot, 'prebuilds', `${process.platform}-${runtimeArch}`, 'pty.node')), true)
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

test('stageNodePty preserves Debug and prebuild fallbacks when Release is absent', () => {
  withTempRoot(root => {
    const srcRoot = join(root, 'source')
    const destRoot = join(root, 'staged')
    const debug = join(srcRoot, 'build', 'Debug')
    const runtimeArch = nodeRuntimeArch(process.arch)
    const prebuild = join(srcRoot, 'prebuilds', `${process.platform}-${runtimeArch}`)
    mkdirSync(srcRoot, { recursive: true })
    writePackageFixture(srcRoot)
    writePayload(debug, { spawnHelper: process.platform === 'darwin' })
    writePayload(prebuild, { spawnHelper: process.platform === 'darwin' })

    stageNodePty({ srcRoot, destRoot })

    assert.equal(existsSync(join(destRoot, 'build', 'Debug', 'pty.node')), true)
    assert.equal(existsSync(join(destRoot, 'prebuilds', `${process.platform}-${runtimeArch}`, 'pty.node')), true)
  })
})

test('wrong-architecture Release cannot suppress a valid target prebuild', () => {
  withTempRoot(root => {
    const release = join(root, 'build', 'Release')
    const prebuild = join(root, 'prebuilds', 'linux-x64')
    writePayload(release, { platform: 'linux', arch: 'arm64' })
    writePayload(prebuild, { platform: 'linux', arch: 'x64' })

    assert.deepEqual(findNodePtyNativePayloads({ root, platform: 'linux', arch: 'x64', includeBuild: true }), [
      prebuild
    ])
    assert.equal(nativeBinaryMatchesTarget(join(release, 'pty.node'), { platform: 'linux', arch: 'x64' }), false)
  })
})

test('stageNodePty makes a Darwin helper executable', () => {
  withTempRoot(root => {
    const srcRoot = join(root, 'source')
    const destRoot = join(root, 'staged')
    const prebuild = join(srcRoot, 'prebuilds', 'darwin-x64')
    mkdirSync(srcRoot, { recursive: true })
    writePackageFixture(srcRoot)
    writePayload(prebuild, {
      platform: 'darwin',
      arch: 'x64',
      spawnHelper: true,
      helperMode: 0o644
    })

    stageNodePty({ platform: 'darwin', arch: 'x64', srcRoot, destRoot })

    assert.notEqual(statSync(join(destRoot, 'prebuilds', 'darwin-x64', 'spawn-helper')).mode & 0o111, 0)
  })
})

test('copyDirRecursive preserves nested binary bytes without recursive cpSync', () => {
  withTempRoot(root => {
    const src = join(root, 'src')
    const dest = join(root, 'dest')
    mkdirSync(join(src, 'conpty'), { recursive: true })
    writeFileSync(join(src, 'conpty', 'conpty.dll'), Buffer.from([0, 1, 2, 255]))

    copyDirRecursive(src, dest)

    assert.deepEqual(readFileSync(join(dest, 'conpty', 'conpty.dll')), Buffer.from([0, 1, 2, 255]))
  })
})

test('stage-native-deps contains no recursive cpSync call', () => {
  const source = readFileSync(new URL('./stage-native-deps.mjs', import.meta.url), 'utf8')
  assert.equal(usesRecursiveCpSync(source), false)
})

test(
  'rebuildNodePty replaces an existing Linux payload through the npm workspace',
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
      writePayload(join(buildDir, 'Release'), { platform: 'linux', arch: process.arch })
      const stale = readFileSync(join(buildDir, 'Release', 'pty.node'))
      assert.notEqual(findNodePtyNativePayload(), undefined)
      assert.equal(await rebuildNodePty(), true)
      assert.equal(existsSync(join(buildDir, 'Release', 'pty.node')), true)
      assert.notDeepEqual(readFileSync(join(buildDir, 'Release', 'pty.node')), stale)
    } finally {
      rmSync(buildDir, { recursive: true, force: true })
      if (hadBuild) renameSync(backupDir, buildDir)
    }
  }
)
