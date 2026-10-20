import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test, vi } from 'vitest'

import { resolveSourceInstallationBackend } from './source-backend'
import { createValidatedSourceResolver, resolveUpdateRoot } from './update-root'

afterEach((): void => {
  vi.unstubAllEnvs()
})

function sourceTree(root: string): void {
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
  fs.writeFileSync(path.join(root, 'hermes_cli', '_launchers.py'), '')
}

function publishLauncher(dir: string, reported: string): string {
  fs.mkdirSync(dir, { recursive: true })
  const launcher: string = path.join(dir, 'hermes')
  fs.writeFileSync(launcher, `#!/bin/sh\nprintf '%s\\n' 'Install directory: ${reported}'\n`, { mode: 0o755 })

  return launcher
}

test.skipIf(process.platform === 'win32')(
  'an isolated POSIX source install validated by backend resolution is the update root',
  async (): Promise<void> => {
    const base: string = fs.realpathSync(fs.mkdtempSync(path.join(os.tmpdir(), 'update-root-')))

    try {
      const home: string = path.join(base, 'home')
      const alternateRoot: string = path.join(home, 'src', 'hermes-agent')
      sourceTree(alternateRoot)
      const launcher: string = publishLauncher(path.join(home, '.local', 'bin'), alternateRoot)
      vi.stubEnv('HOME', home)
      vi.stubEnv('PATH', '/usr/bin:/bin:/usr/sbin:/sbin')

      assert.equal(fs.existsSync(launcher), true)

      const backend = await resolveSourceInstallationBackend(alternateRoot, ['serve'], {
        isWindows: false,
        hermesHome: path.join(home, '.hermes')
      })

      assert.equal(backend?.root, alternateRoot)
      assert.equal(
        resolveUpdateRoot({
          isPackaged: true,
          sourceRepoRoot: path.join(base, 'dev'),
          activeHermesRoot: path.join(home, '.hermes', 'hermes-agent'),
          validatedSourceInstallRoot: backend?.root,
          isHermesSourceRoot: root => fs.existsSync(path.join(root, 'hermes_cli', 'main.py')),
          isGitCheckout: root => fs.existsSync(path.join(root, '.git'))
        }),
        alternateRoot
      )
    } finally {
      fs.rmSync(base, { recursive: true, force: true })
    }
  }
)

test('validated source identity clears after a newer non-source result', async (): Promise<void> => {
  let outcome: 'source' | 'canonical' | 'bootstrap' | 'throw' = 'source'
  let settle: (() => void) | undefined
  let reject: ((error: Error) => void) | undefined

  const resolver = createValidatedSourceResolver(async (_args, select): Promise<string> => {
    if (outcome === 'source') {
      select('/tmp/hermes-source')

      return 'source'
    }

    await new Promise<void>((resolve, fail): void => {
      settle = resolve
      reject = fail
    })

    if (outcome === 'throw') {
      throw new Error('probe failed')
    }

    return outcome
  })

  await resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-source')

  outcome = 'canonical'
  const canonical = resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-source')
  assert.ok(settle)
  settle()
  await canonical
  assert.equal(resolver.current(), undefined)

  outcome = 'source'
  await resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-source')

  outcome = 'bootstrap'
  const bootstrap = resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-source')
  assert.ok(settle)
  settle()
  await bootstrap
  assert.equal(resolver.current(), undefined)

  outcome = 'source'
  await resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-source')

  outcome = 'throw'
  const failed = resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-source')
  assert.ok(reject)
  reject(new Error('probe failed'))
  await assert.rejects(failed, /probe failed/)
  assert.equal(resolver.current(), undefined)
})

test('validated source identity publishes only when the attempt selects a source root', async (): Promise<void> => {
  const resolver = createValidatedSourceResolver(async (_args, select): Promise<string> => {
    select('/tmp/hermes-alternate')

    return 'source'
  })

  await resolver([])
  assert.equal(resolver.current(), '/tmp/hermes-alternate')
})

test('an older backend attempt cannot overwrite a newer selected identity', async (): Promise<void> => {
  const pending: Array<{
    select: (root: string) => void
    resolve: () => void
    reject: (error: Error) => void
  }> = []

  const resolver = createValidatedSourceResolver(
    async (_args, select): Promise<string> =>
      new Promise((resolve, reject): void => {
        pending.push({ select, resolve: (): void => resolve('done'), reject })
      })
  )

  const older = resolver([])
  const newer = resolver([])
  pending[1].reject(new Error('newer probe failed'))
  await assert.rejects(newer, /newer probe failed/)
  pending[0].select('/tmp/hermes-older')
  pending[0].resolve()
  await older

  assert.equal(resolver.current(), undefined)
})

test('override and development roots keep precedence over a validated alternate root', (): void => {
  const roots = {
    overrideRoot: '/tmp/hermes-override',
    sourceRepoRoot: '/tmp/hermes-dev',
    activeHermesRoot: '/tmp/hermes-active',
    validatedSourceInstallRoot: '/tmp/hermes-alternate'
  }

  const sourceRoots = new Set(Object.values(roots))

  const gitRoots = new Set([
    roots.overrideRoot,
    roots.sourceRepoRoot,
    roots.activeHermesRoot,
    roots.validatedSourceInstallRoot
  ])

  const options = {
    ...roots,
    isHermesSourceRoot: (root: string): boolean => sourceRoots.has(root),
    isGitCheckout: (root: string): boolean => gitRoots.has(root)
  }

  assert.equal(resolveUpdateRoot({ ...options, isPackaged: true }), roots.overrideRoot)
  assert.equal(resolveUpdateRoot({ ...options, isPackaged: true, overrideRoot: undefined }), roots.activeHermesRoot)
  assert.equal(resolveUpdateRoot({ ...options, isPackaged: false, overrideRoot: undefined }), roots.sourceRepoRoot)
})

test('an invalid validated root does not displace the canonical source root', (): void => {
  const activeHermesRoot = '/tmp/hermes-active'

  assert.equal(
    resolveUpdateRoot({
      isPackaged: true,
      sourceRepoRoot: '/tmp/hermes-dev',
      activeHermesRoot,
      validatedSourceInstallRoot: '/tmp/unrelated',
      isHermesSourceRoot: root => root === activeHermesRoot,
      isGitCheckout: () => false
    }),
    activeHermesRoot
  )
})
