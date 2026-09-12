import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, expect, it, vi } from 'vitest'

import { checkCheckoutUpdates, type CheckoutCheckDeps } from './checkout-check'

const roots: string[] = []
afterEach((): void => {
  for (const root of roots.splice(0)) {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

function fixture(): CheckoutCheckDeps {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'checkout-check-'))
  roots.push(root)

  return {
    writeFileAtomic: (filePath: string, contents: string): void => fs.writeFileSync(filePath, contents),
    updateCheckCachePath: path.join(root, 'cache.json'),
    isGitCheckout: (): boolean => true,
    readCanonicalInstallStamp: (): null => null,
    readDesktopUpdateConfig: (): { branch: string } => ({ branch: 'main' }),
    resolveUpdateRoot: (): string => root,
    resolveHealedBranch: async (_root: string, branch: string): Promise<string> => branch,
    getOriginUrl: async (): Promise<string> => 'git@github.com:NousResearch/hermes-agent.git',
    runGit: vi.fn(async (args: string[]): Promise<{ code: number; stdout: string; stderr: string }> => {
      const key = args.join(' ')

      if (key === 'rev-parse HEAD') {
        return { code: 0, stdout: 'a'.repeat(40), stderr: '' }
      }

      if (key === 'rev-parse --abbrev-ref HEAD') {
        return { code: 0, stdout: 'main', stderr: '' }
      }

      if (key === 'status --porcelain') {
        return { code: 0, stdout: '', stderr: '' }
      }

      throw new Error(`Unexpected git operation: ${key}`)
    }),
    readSourceUpdate: async (): Promise<{ channel: 'main' }> => ({ channel: 'main' }),
    fetchGitHubApi: vi.fn(async (url: string): Promise<unknown> =>
      url.includes('/commits/') ? 'b'.repeat(40) : { ahead_by: 3, commits: [] }
    ),
    rememberLog: vi.fn()
  }
}

it('uses API checks and a disk cache while a forced check bypasses the cache', async (): Promise<void> => {
  const deps = fixture()
  const first = await checkCheckoutUpdates(deps)
  expect(first.behind).toBe(3)
  expect(first.updateAvailable).toBe(true)
  expect(deps.fetchGitHubApi).toHaveBeenCalledTimes(2)
  expect(await checkCheckoutUpdates(deps)).toEqual(first)
  expect(deps.fetchGitHubApi).toHaveBeenCalledTimes(2)
  await checkCheckoutUpdates(deps, { force: true })
  expect(deps.fetchGitHubApi).toHaveBeenCalledTimes(4)
})

it('does not report locally ahead commits as an update', async (): Promise<void> => {
  const deps = fixture()
  deps.fetchGitHubApi = async (url: string): Promise<unknown> =>
    url.includes('/commits/') ? 'b'.repeat(40) : { ahead_by: 0 }
  expect(await checkCheckoutUpdates(deps)).toMatchObject({ behind: 0, updateAvailable: false, commits: [] })
})

it('keeps an unknown count when the compare API fails', async (): Promise<void> => {
  const deps = fixture()

  deps.fetchGitHubApi = async (url: string): Promise<unknown> => {
    if (url.includes('/commits/')) {
      return 'b'.repeat(40)
    }

    throw new Error('rate limited')
  }

  expect(await checkCheckoutUpdates(deps)).toMatchObject({ behind: null, updateAvailable: true })
})

it('uses only ls-remote for non-GitHub network checks', async (): Promise<void> => {
  const deps = fixture()
  const localGit = deps.runGit
  deps.getOriginUrl = async (): Promise<string> => 'https://git.example/repo.git'
  deps.runGit = vi.fn(
    async (args: string[], options: { cwd?: string }): Promise<{ code: number; stdout: string; stderr: string }> => {
      if (args[0] === 'ls-remote') {
        return { code: 0, stdout: `${'b'.repeat(40)}\trefs/heads/main`, stderr: '' }
      }

      if (args[0] === 'cat-file') {
        return { code: 1, stdout: '', stderr: '' }
      }

      return localGit(args, options)
    }
  )
  expect(await checkCheckoutUpdates(deps)).toMatchObject({ behind: null, updateAvailable: true })
  expect(deps.fetchGitHubApi).not.toHaveBeenCalled()
})
