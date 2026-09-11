/**
 * Tests for electron/branch-healer.ts — self-healing logic for the desktop
 * updater branch-pin (#105042).
 *
 * Run with: node --test apps/desktop/electron/branch-healer.test.ts
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { resolveHealedBranch, type BranchHealerDeps } from './branch-healer.ts'

function createMockDeps(overrides: Partial<BranchHealerDeps> = {}): {
  deps: BranchHealerDeps
  logs: string[]
  writtenConfig: () => { branch: string } | null
} {
  const logs: string[] = []
  let writtenConfig: { branch: string } | null = null

  const deps: BranchHealerDeps = {
    runGit: async () => ({ code: 0, stdout: '', stderr: '' }),
    getOriginUrl: async () => 'https://github.com/NousResearch/hermes-agent.git',
    isOfficialSshRemote: () => false,
    officialRepoHttpsUrl: 'https://github.com/NousResearch/hermes-agent.git',
    rememberLog: (msg: string) => logs.push(msg),
    readDesktopUpdateConfig: () => ({ branch: 'feature' }),
    writeDesktopUpdateConfig: (cfg: { branch: string }) => {
      writtenConfig = cfg
    },
    ...overrides
  }

  return { deps, logs, writtenConfig: () => writtenConfig }
}

test('resolveHealedBranch: returns main or falsy branch immediately without git calls', async () => {
  let gitCalled = false
  const { deps } = createMockDeps({
    runGit: async () => {
      gitCalled = true
      return { code: 0, stdout: '', stderr: '' }
    }
  })

  assert.equal(await resolveHealedBranch(deps, '/fake/root', 'main'), 'main')
  assert.equal(await resolveHealedBranch(deps, '/fake/root', ''), 'main')
  assert.equal(gitCalled, false)
})

test('resolveHealedBranch: keeps branch when remote still publishes it (ls-remote exit 0)', async () => {
  const { deps, writtenConfig } = createMockDeps({
    runGit: async (args: string[]) => {
      if (args[0] === 'ls-remote') {
        return { code: 0, stdout: 'abc1234 refs/heads/feature\n', stderr: '' }
      }
      return { code: 0, stdout: '', stderr: '' }
    }
  })

  const result = await resolveHealedBranch(deps, '/fake/root', 'feature')
  assert.equal(result, 'feature')
  assert.equal(writtenConfig(), null)
})

test('resolveHealedBranch: keeps branch on transient network failure (ls-remote exit 128)', async () => {
  const { deps, writtenConfig } = createMockDeps({
    runGit: async (args: string[]) => {
      if (args[0] === 'ls-remote') {
        return { code: 128, stdout: '', stderr: 'fatal: could not resolve host' }
      }
      return { code: 0, stdout: '', stderr: '' }
    }
  })

  const result = await resolveHealedBranch(deps, '/fake/root', 'feature')
  assert.equal(result, 'feature')
  assert.equal(writtenConfig(), null)
})

test('resolveHealedBranch: keeps branch when exit 2 but branch is purely local (no remote-tracking ref or upstream)', async () => {
  const { deps, writtenConfig } = createMockDeps({
    runGit: async (args: string[]) => {
      if (args[0] === 'ls-remote') {
        return { code: 2, stdout: '', stderr: '' }
      }
      if (args[0] === 'rev-parse') {
        // neither refs/remotes/origin/feature nor feature@{upstream} exist
        return { code: 1, stdout: '', stderr: 'fatal: Needed a single revision' }
      }
      return { code: 0, stdout: '', stderr: '' }
    }
  })

  const result = await resolveHealedBranch(deps, '/fake/root', 'hermes-local')
  assert.equal(result, 'hermes-local')
  assert.equal(writtenConfig(), null)
})

test('resolveHealedBranch: keeps branch when exit 2 and remote ref existed, but branch carries unmerged commits', async () => {
  const { deps, writtenConfig } = createMockDeps({
    runGit: async (args: string[]) => {
      if (args[0] === 'ls-remote') {
        return { code: 2, stdout: '', stderr: '' }
      }
      if (args[0] === 'rev-parse') {
        // remote tracking ref existed
        return { code: 0, stdout: 'abc123\n', stderr: '' }
      }
      if (args[0] === 'rev-list') {
        // 3 unmerged commits
        return { code: 0, stdout: '3\n', stderr: '' }
      }
      return { code: 0, stdout: '', stderr: '' }
    }
  })

  const result = await resolveHealedBranch(deps, '/fake/root', 'feature-unmerged')
  assert.equal(result, 'feature-unmerged')
  assert.equal(writtenConfig(), null)
})

test('resolveHealedBranch: heals to main and persists config when exit 2, remote ref existed, and 0 unmerged commits', async () => {
  const { deps, writtenConfig } = createMockDeps({
    runGit: async (args: string[]) => {
      if (args[0] === 'ls-remote') {
        return { code: 2, stdout: '', stderr: '' }
      }
      if (args[0] === 'rev-parse') {
        // remote tracking ref existed
        return { code: 0, stdout: 'abc123\n', stderr: '' }
      }
      if (args[0] === 'rev-list') {
        // 0 unmerged commits (fully merged into main)
        return { code: 0, stdout: '0\n', stderr: '' }
      }
      return { code: 0, stdout: '', stderr: '' }
    }
  })

  const result = await resolveHealedBranch(deps, '/fake/root', 'bb/gui')
  assert.equal(result, 'main')
  assert.deepEqual(writtenConfig(), { branch: 'main' })
})
