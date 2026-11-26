import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import {
  createCltGitCache,
  isUsableMacGitCandidate,
  probeXcodeSelectDeveloperDir,
  resolvePosixGitBinary,
  runOptionalGit
} from './mac-clt-git'

const tempDirs: string[] = []

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

function makeTempDir(prefix: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix))

  tempDirs.push(dir)

  return dir
}

function makeExecutable(filePath: string, body = '#!/usr/bin/env bash\nexit 0\n'): string {
  fs.mkdirSync(path.dirname(filePath), { recursive: true })
  fs.writeFileSync(filePath, body, { encoding: 'utf8', mode: 0o755 })
  fs.chmodSync(filePath, 0o755)

  return filePath
}

test('uninstalled macOS CLT stub is skipped without probing and resolves to null', () => {
  const dir = makeTempDir('hermes-mac-clt-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const probed: string[] = []

  const selected = resolvePosixGitBinary({
    candidates: [stub],
    platform: 'darwin',
    env: { PATH: path.dirname(stub) },
    appleStubPath: stub,
    developerDirProbe: () => '',
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: candidate => {
      probed.push(candidate)

      return false
    }
  })

  assert.equal(selected, null)
  assert.deepEqual(probed, [])
})

test('symlink to macOS CLT stub is also blocked before binaryRuns', () => {
  const dir = makeTempDir('hermes-mac-clt-link-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const aliasDir = path.join(dir, 'alias')
  const alias = path.join(aliasDir, 'git')

  fs.mkdirSync(aliasDir, { recursive: true })
  fs.symlinkSync(stub, alias)

  const probed: string[] = []

  const selected = resolvePosixGitBinary({
    candidates: [alias],
    platform: 'darwin',
    env: { PATH: aliasDir },
    appleStubPath: stub,
    developerDirProbe: () => '',
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: candidate => {
      probed.push(candidate)

      return false
    }
  })

  assert.equal(selected, null)
  assert.deepEqual(probed, [])
})

test('Homebrew git ahead of CLT stub wins without calling xcode-select or probing the stub', () => {
  const dir = makeTempDir('hermes-mac-clt-brew-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const homebrew = makeExecutable(path.join(dir, 'homebrew', 'bin', 'git'))
  const probed: string[] = []

  const selected = resolvePosixGitBinary({
    candidates: [homebrew, stub],
    platform: 'darwin',
    env: { PATH: `${path.dirname(homebrew)}:${path.dirname(stub)}` },
    appleStubPath: stub,
    developerDirProbe: () => {
      throw new Error('xcode-select must not run when Homebrew git already runs')
    },
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: candidate => {
      probed.push(candidate)

      return candidate === homebrew
    }
  })

  assert.equal(selected, homebrew)
  assert.deepEqual(probed, [homebrew])
})

test('Homebrew git after CLT stub is selected while skipping the stub without probing it', () => {
  const dir = makeTempDir('hermes-mac-clt-after-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const homebrew = makeExecutable(path.join(dir, 'homebrew', 'bin', 'git'))
  const probed: string[] = []

  const selected = resolvePosixGitBinary({
    candidates: [stub, homebrew],
    platform: 'darwin',
    env: { PATH: `${path.dirname(stub)}:${path.dirname(homebrew)}` },
    appleStubPath: stub,
    developerDirProbe: () => '',
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: candidate => {
      probed.push(candidate)

      return candidate === homebrew
    }
  })

  assert.equal(selected, homebrew)
  assert.deepEqual(probed, [homebrew])
})

test('Homebrew git missing from a minimal Finder PATH is found in the fallback dirs', () => {
  const dir = makeTempDir('hermes-mac-clt-fallback-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const homebrew = makeExecutable(path.join(dir, 'homebrew', 'bin', 'git'))
  const probed: string[] = []

  const selected = resolvePosixGitBinary({
    candidates: [stub],
    fallbackDirs: [path.dirname(homebrew), path.dirname(stub)],
    platform: 'darwin',
    env: { PATH: path.dirname(stub) },
    appleStubPath: stub,
    developerDirProbe: () => '',
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: candidate => {
      probed.push(candidate)

      return candidate === homebrew
    }
  })

  assert.equal(selected, homebrew)
  assert.deepEqual(probed, [homebrew])
})

test('fallback dirs come after PATH and a dir on both is probed once', () => {
  const dir = makeTempDir('hermes-mac-clt-fallback-order-')
  const broken = makeExecutable(path.join(dir, 'local', 'bin', 'git'))
  const nix = makeExecutable(path.join(dir, 'nix', 'bin', 'git'))
  const homebrew = makeExecutable(path.join(dir, 'homebrew', 'bin', 'git'))
  const probed: string[] = []

  const resolve = (candidates: string[]) =>
    resolvePosixGitBinary({
      candidates,
      fallbackDirs: [path.dirname(broken), path.dirname(homebrew)],
      platform: 'linux',
      env: { PATH: candidates.map(candidate => path.dirname(candidate)).join(':') },
      fileExists: candidate => fs.existsSync(candidate),
      binaryRuns: candidate => {
        probed.push(candidate)

        return candidate !== broken
      }
    })

  assert.equal(resolve([nix]), nix)
  assert.deepEqual(probed.splice(0), [nix])
  assert.equal(resolve([broken]), homebrew)
  assert.deepEqual(probed, [broken, homebrew])
})

test('DEVELOPER_DIR accepts both a Developer directory and an Xcode.app bundle', () => {
  const dir = makeTempDir('hermes-mac-clt-devdir-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const xcodeApp = path.join(dir, 'Xcode.app')
  const emptyApp = path.join(dir, 'Empty.app')

  makeExecutable(path.join(xcodeApp, 'Contents', 'Developer', 'usr', 'bin', 'git'))
  fs.mkdirSync(path.join(emptyApp, 'Contents', 'Developer'), { recursive: true })

  const failProbe = () => {
    throw new Error('xcode-select must not run when DEVELOPER_DIR is set')
  }

  assert.equal(
    isUsableMacGitCandidate(stub, {
      platform: 'darwin',
      env: { DEVELOPER_DIR: xcodeApp },
      appleStubPath: stub,
      developerDirProbe: failProbe,
      cache: createCltGitCache()
    }),
    true
  )
  assert.equal(
    isUsableMacGitCandidate(stub, {
      platform: 'darwin',
      env: { DEVELOPER_DIR: path.join(xcodeApp, 'Contents', 'Developer') },
      appleStubPath: stub,
      developerDirProbe: failProbe,
      cache: createCltGitCache()
    }),
    true
  )
  assert.equal(
    isUsableMacGitCandidate(stub, {
      platform: 'darwin',
      env: { DEVELOPER_DIR: emptyApp },
      appleStubPath: stub,
      developerDirProbe: failProbe,
      cache: createCltGitCache()
    }),
    false
  )
})

test('negative CLT cache expires after TTL while positive cache sticks', () => {
  const dir = makeTempDir('hermes-mac-clt-cache-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))
  const devDir = path.join(dir, 'clt')

  fs.mkdirSync(devDir, { recursive: true })

  let now = 1_000
  let calls = 0
  let installed = false
  const cache = createCltGitCache(5_000)

  const check = () =>
    isUsableMacGitCandidate(stub, {
      platform: 'darwin',
      env: { PATH: path.dirname(stub) },
      appleStubPath: stub,
      developerDirProbe: () => {
        calls += 1

        return installed ? devDir : ''
      },
      cache,
      nowMs: () => now
    })

  assert.equal(check(), false)
  assert.equal(calls, 1)

  now = 5_000
  assert.equal(check(), false)
  assert.equal(calls, 1)

  makeExecutable(path.join(devDir, 'usr', 'bin', 'git'))
  installed = true
  now = 6_000
  assert.equal(check(), true)
  assert.equal(calls, 2)

  now = 999_999
  assert.equal(check(), true)
  assert.equal(calls, 2)
})

test('non-darwin hosts never run the CLT probe', () => {
  const dir = makeTempDir('hermes-mac-clt-linux-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))

  const selected = resolvePosixGitBinary({
    candidates: [stub],
    platform: 'linux',
    env: { PATH: path.dirname(stub), DEVELOPER_DIR: path.join(dir, 'missing') },
    appleStubPath: stub,
    developerDirProbe: () => {
      throw new Error('CLT probe must only run on darwin')
    },
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: () => true
  })

  assert.equal(selected, stub)
})

test('unset PATH on darwin with only the CLT stub resolves to null instead of bare git', () => {
  const dir = makeTempDir('hermes-mac-clt-unset-path-')
  const stub = makeExecutable(path.join(dir, 'usr_bin', 'git'))

  const selected = resolvePosixGitBinary({
    candidates: [],
    platform: 'darwin',
    env: {},
    appleStubPath: stub,
    developerDirProbe: () => '',
    cache: createCltGitCache(),
    fileExists: candidate => fs.existsSync(candidate),
    binaryRuns: () => false
  })

  assert.equal(selected, null)
})

test.skipIf(process.platform === 'win32')('xcode-select probe enforces its timeout on a slow response', () => {
  const dir = makeTempDir('hermes-mac-clt-timeout-')
  const devDir = path.join(dir, 'developer')

  makeExecutable(path.join(devDir, 'usr', 'bin', 'git'))

  const makeSelect = (name: string, delayMs: number) =>
    makeExecutable(
      path.join(dir, name),
      `#!/usr/bin/env bash\nexec ${JSON.stringify(process.execPath)} -e ${JSON.stringify(
        `setTimeout(() => console.log(${JSON.stringify(devDir)}), ${delayMs})`
      )}\n`
    )

  const fast = probeXcodeSelectDeveloperDir(process.env, {
    xcodeSelect: makeSelect('xcode-select-fast', 0),
    timeoutMs: 30_000
  })

  assert.equal(fast, devDir)

  const slow = probeXcodeSelectDeveloperDir(process.env, {
    xcodeSelect: makeSelect('xcode-select-slow', 2_000),
    timeoutMs: 250
  })

  assert.equal(slow, '')
})

test('runOptionalGit short-circuits without spawning when gitBin is null', async () => {
  let called = 0

  const result = await runOptionalGit(null, ['rev-parse', 'HEAD'], {}, async () => {
    called += 1

    return { code: 0, stdout: 'ok', stderr: '' }
  })

  assert.equal(called, 0)
  assert.deepEqual(result, { code: 1, stdout: '', stderr: 'git is not available' })
})
