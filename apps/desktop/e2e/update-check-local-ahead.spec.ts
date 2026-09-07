import { execFileSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as path from 'node:path'
import { pathToFileURL } from 'node:url'

import { buildAppEnv, createSandbox, launchDesktop } from './fixtures'
import { allowErrorBanners, expect, test } from './test'

const OFFICIAL_HTTPS_REMOTE = 'https://github.com/NousResearch/hermes-agent.git'
const OFFICIAL_SSH_REMOTE = 'git@github.com:NousResearch/hermes-agent.git'

interface UpdateStatus {
  behind: number | null
  branch: string
  currentSha: string
  error?: string
  supported: boolean
  targetSha: string
  updateAvailable: boolean
}

interface UpdateWindow {
  hermesDesktop?: {
    updates?: {
      check: () => Promise<UpdateStatus>
    }
  }
}

function runGit(cwd: string, args: string[]): string {
  return execFileSync('git', args, { cwd, encoding: 'utf8', timeout: 10_000 }).trim()
}

function createLocalAheadCheckout(root: string) {
  const remote = path.join(root, 'remote.git')
  const seed = path.join(root, 'seed')
  const checkout = path.join(root, 'checkout')

  fs.mkdirSync(seed)
  runGit(root, ['init', '--bare', remote])
  runGit(seed, ['init', '--quiet'])
  runGit(seed, ['config', 'commit.gpgSign', 'false'])
  runGit(seed, ['config', 'core.hooksPath', '.git/no-hooks'])
  runGit(seed, ['config', 'user.name', 'Hermes Test'])
  runGit(seed, ['config', 'user.email', 'hermes@example.invalid'])
  runGit(seed, ['commit', '--allow-empty', '-m', 'remote tip'])
  runGit(seed, ['branch', '-M', 'main'])
  runGit(seed, ['remote', 'add', 'origin', pathToFileURL(remote).href])
  runGit(seed, ['push', '--quiet', 'origin', 'main'])

  runGit(root, ['clone', '--quiet', '--depth', '1', '--branch', 'main', pathToFileURL(remote).href, checkout])
  runGit(checkout, ['config', 'commit.gpgSign', 'false'])
  runGit(checkout, ['config', 'core.hooksPath', '.git/no-hooks'])
  runGit(checkout, ['config', 'user.name', 'Hermes Test'])
  runGit(checkout, ['config', 'user.email', 'hermes@example.invalid'])

  const targetSha = runGit(checkout, ['rev-parse', 'HEAD'])

  runGit(checkout, ['commit', '--allow-empty', '-m', 'carried local commit'])
  runGit(checkout, ['remote', 'set-url', 'origin', OFFICIAL_SSH_REMOTE])

  return {
    checkout,
    currentSha: runGit(checkout, ['rev-parse', 'HEAD']),
    remoteUrl: pathToFileURL(remote).href,
    targetSha
  }
}

test('a carried local commit is current through the real update-check IPC path', async () => {
  allowErrorBanners()

  const sandbox = createSandbox('update-local-ahead')
  const fixture = createLocalAheadCheckout(sandbox.root)

  const env = buildAppEnv(sandbox, {
    GIT_CONFIG_COUNT: '1',
    GIT_CONFIG_KEY_0: `url.${fixture.remoteUrl}.insteadOf`,
    GIT_CONFIG_VALUE_0: OFFICIAL_HTTPS_REMOTE,
    HERMES_DESKTOP_BOOT_FAKE_ERROR: 'E2E update-check spec: local backend intentionally not started',
    HERMES_DESKTOP_HERMES_ROOT: fixture.checkout
  })

  const { app, page } = await launchDesktop(env)

  try {
    await page.waitForFunction(() => Boolean((window as unknown as UpdateWindow).hermesDesktop?.updates), undefined, {
      timeout: 60_000
    })

    const status = await page.evaluate(() => {
      const updates = (window as unknown as UpdateWindow).hermesDesktop?.updates

      if (!updates) {
        throw new Error('desktop update bridge unavailable')
      }

      return updates.check()
    })

    expect(status).toMatchObject({
      behind: 0,
      branch: 'main',
      currentSha: fixture.currentSha,
      supported: true,
      targetSha: fixture.targetSha,
      updateAvailable: false
    })
    expect(status.error).toBeUndefined()
    expect(status.currentSha).not.toBe(status.targetSha)
  } finally {
    await app.close().catch(() => undefined)
    sandbox.cleanup()
  }
})
