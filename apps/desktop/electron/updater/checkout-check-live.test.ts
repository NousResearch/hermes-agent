import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as http from 'node:http'
import type { AddressInfo } from 'node:net'
import * as os from 'node:os'
import * as path from 'node:path'

import { expect, it } from 'vitest'

import { checkCheckoutUpdates, type CheckoutCheckDeps } from './checkout-check'

it('checks a real linked worktree through HTTP and reuses the disk cache until forced or HEAD changes', async (): Promise<void> => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-worktree-'))
  const checkout = path.join(root, 'source')
  const worktree = path.join(root, 'worktree')
  const requests: string[] = []

  const server = http.createServer((request: http.IncomingMessage, response: http.ServerResponse): void => {
    requests.push(request.url ?? '')
    response.end(targetSha)
  })

  let targetSha = ''

  function git(args: string[], cwd: string = checkout): string {
    return execFileSync('git', ['-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', '-c', 'commit.gpgsign=false', ...args], {
      cwd,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe']
    }).trim()
  }

  try {
    fs.mkdirSync(checkout)
    git(['init', '--initial-branch=main'])
    git(['commit', '--allow-empty', '-m', 'initial'])
    git(['remote', 'add', 'origin', 'https://github.com/example/test.git'])
    git(['worktree', 'add', '--detach', worktree])
    targetSha = git(['rev-parse', 'HEAD'])
    await new Promise<void>((resolve: () => void): void => { server.listen(0, '127.0.0.1', resolve) })
    const address = server.address() as AddressInfo

    const deps: CheckoutCheckDeps = {
      updateCheckCachePath: path.join(root, 'cache.json'),
      writeFileAtomic: (filePath: string, contents: string): void => {
        fs.writeFileSync(`${filePath}.tmp`, contents)
        fs.renameSync(`${filePath}.tmp`, filePath)
      },
      isGitCheckout: (directory: string): boolean => fs.existsSync(path.join(directory, '.git')),
      readCanonicalInstallStamp: (): null => null,
      readDesktopUpdateConfig: (): { branch: string } => ({ branch: 'main' }),
      resolveUpdateRoot: (): string => worktree,
      resolveHealedBranch: async (_directory: string, branch: string): Promise<string> => branch,
      getOriginUrl: async (directory: string): Promise<string> => git(['remote', 'get-url', 'origin'], directory),
      runGit: async (args: string[], options?: { cwd?: string }): Promise<{ code: number; stdout: string; stderr: string }> => {
        // A passive check must not use git to contact the configured remote.
        expect(['rev-parse', 'status']).toContain(args[0])

        return { code: 0, stdout: git(args, options?.cwd), stderr: '' }
      },
      readSourceUpdate: async (): Promise<{ channel: 'main' }> => ({ channel: 'main' }),
    fetchGitHubApi: async (url: string, accept?: string): Promise<unknown> => {
        const parsed = new URL(url)
        expect(parsed.hostname).toBe('api.github.com')

        const response = await fetch(`http://127.0.0.1:${address.port}${parsed.pathname}`, {
          headers: { Accept: accept ?? 'application/json' }
        })

        return response.text()
      },
      rememberLog: (message: unknown): void => { throw new Error(String(message)) }
    }

    expect(fs.statSync(path.join(worktree, '.git')).isFile()).toBe(true)
    expect(await checkCheckoutUpdates(deps)).toMatchObject({ supported: true, currentSha: targetSha, updateAvailable: false })
    expect(requests).toHaveLength(1)
    await checkCheckoutUpdates({ ...deps })
    expect(requests).toHaveLength(1)
    await checkCheckoutUpdates(deps, { force: true })
    expect(requests).toHaveLength(2)
    git(['commit', '--allow-empty', '-m', 'advance'], worktree)
    targetSha = git(['rev-parse', 'HEAD'], worktree)
    expect(await checkCheckoutUpdates(deps)).toMatchObject({ currentSha: targetSha, updateAvailable: false })
    expect(requests).toHaveLength(3)
  } finally {
    server.closeAllConnections()
    await new Promise<void>((resolve: () => void): void => { server.close((): void => resolve()) })
    fs.rmSync(root, { recursive: true, force: true })
  }
})
