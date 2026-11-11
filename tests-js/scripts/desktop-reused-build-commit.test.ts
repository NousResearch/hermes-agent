import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { expect, test } from 'vitest'

import { assertChatCommit } from './desktop-chat-smoke.ts'
import { reusedDesktopBuildCommit } from './desktop-smoke-process.ts'

function git(repo: string, ...args: string[]): string {
  const result = spawnSync('git', ['-C', repo, ...args], {
    encoding: 'utf8',
    env: { ...process.env, GIT_AUTHOR_NAME: 't', GIT_AUTHOR_EMAIL: 't@t.invalid', GIT_COMMITTER_NAME: 't', GIT_COMMITTER_EMAIL: 't@t.invalid' },
  })
  if (result.status !== 0) { throw new Error(result.stderr) }

  return result.stdout.trim()
}

function commitFile(repo: string, name: string): string {
  fs.writeFileSync(path.join(repo, name), `${name}\n`)
  git(repo, 'add', name)
  git(repo, 'commit', '-q', '-m', name)

  return git(repo, 'rev-parse', 'HEAD')
}

test('a reused packaged app may report its build commit only across the synthetic NEXT commit', (): void => {
  const repo = fs.mkdtempSync(path.join(os.tmpdir(), 'reused-build-'))

  try {
    git(repo, 'init', '-q')
    const head = commitFile(repo, 'app.ts')
    const next = commitFile(repo, '.hermes-e2e-next')
    const app = { appVersion: '0.0.0', commit: head, hermesRoot: repo, platform: process.platform }

    // HEAD -> NEXT: only the marker changed, so the app built at HEAD stands for NEXT.
    const reused = reusedDesktopBuildCommit(repo, 'source', next)
    expect(reused).toBe(head)
    expect((): void => assertChatCommit(app, next, next, reused)).not.toThrow()

    // A commit that changes anything else gives no allowance: a stale app stays a failure.
    const real = commitFile(repo, 'feature.ts')
    expect(reusedDesktopBuildCommit(repo, 'source', real)).toBeUndefined()
    expect((): void => assertChatCommit({ ...app, commit: next }, real, real, reusedDesktopBuildCommit(repo, 'source', real))).toThrow('does not equal expected')
  } finally { fs.rmSync(repo, { recursive: true, force: true }) }
})
