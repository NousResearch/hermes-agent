const test = require('node:test')
const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')

const {
  MIMALLOC_ENV,
  buildAllocatorEnv,
  mimallocLibBasename,
  resolveMimallocLibPath
} = require('./allocator-env.cjs')

test('mimallocLibBasename picks platform-specific library names', () => {
  assert.equal(mimallocLibBasename('darwin', 'arm64'), 'libmimalloc.dylib')
  assert.equal(mimallocLibBasename('linux', 'x64'), 'libmimalloc.so')
  assert.equal(mimallocLibBasename('win32', 'x64'), 'mimalloc-redirect.dll')
})

test('buildAllocatorEnv returns empty on Windows and when disabled', () => {
  assert.deepEqual(buildAllocatorEnv({ enabled: false, platform: 'darwin' }), {})
  assert.deepEqual(buildAllocatorEnv({ enabled: true, platform: 'win32' }), {})
})

test('buildAllocatorEnv injects preload vars when library exists', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-mimalloc-'))
  const libDir = path.join(tmp, 'build', 'native-deps', 'mimalloc', 'darwin-arm64')
  fs.mkdirSync(libDir, { recursive: true })
  const libPath = path.join(libDir, 'libmimalloc.dylib')
  fs.writeFileSync(libPath, 'fake')

  const env = buildAllocatorEnv({
    enabled: true,
    appRoot: tmp,
    platform: 'darwin',
    arch: 'arm64'
  })

  assert.equal(env.DYLD_INSERT_LIBRARIES, libPath)
  assert.equal(env.MIMALLOC_PURGE_DELAY, MIMALLOC_ENV.MIMALLOC_PURGE_DELAY)
  assert.equal(env.LD_PRELOAD, undefined)
})

test('resolveMimallocLibPath prefers resourcesPath over appRoot', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-mimalloc-'))
  const resourcesLib = path.join(tmp, 'native-deps', 'mimalloc', 'linux-x64', 'libmimalloc.so')
  fs.mkdirSync(path.dirname(resourcesLib), { recursive: true })
  fs.writeFileSync(resourcesLib, 'fake')

  const resolved = resolveMimallocLibPath({
    resourcesPath: tmp,
    appRoot: '/missing',
    platform: 'linux',
    arch: 'x64'
  })

  assert.equal(resolved, resourcesLib)
})
