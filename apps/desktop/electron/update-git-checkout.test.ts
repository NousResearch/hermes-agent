import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import { isGitCheckout } from './update-git-checkout'

const tempDirs: string[] = []

function tempRoot(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-update-git-checkout-'))
  tempDirs.push(root)

  return root
}

afterEach(() => {
  for (const root of tempDirs.splice(0)) {
    fs.rmSync(root, { force: true, recursive: true })
  }
})

test('accepts a normal checkout whose .git entry is a directory', () => {
  const root = tempRoot()
  fs.mkdirSync(path.join(root, '.git'))

  assert.equal(isGitCheckout(root), true)
})

test('accepts a linked worktree whose .git entry is a gitfile', () => {
  const root = tempRoot()
  fs.writeFileSync(path.join(root, '.git'), 'gitdir: /external/git-metadata/hermes-agent.git\n')

  assert.equal(isGitCheckout(root), true)
})

test('rejects a directory without a .git entry', () => {
  assert.equal(isGitCheckout(tempRoot()), false)
})
