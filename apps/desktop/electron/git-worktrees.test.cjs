'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const { resolveWorktree, worktreesForIpc } = require('./git-worktrees.cjs')

function mkTmpDir() {
  return fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-worktrees-'))
}

// ── resolveWorktree ────────────────────────────────────────────────────

test('resolveWorktree returns null for a path outside any git repo', (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  assert.equal(resolveWorktree(root), null)
})

test('resolveWorktree identifies a main worktree (.git directory)', (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const gitDir = path.join(root, '.git')
  fs.mkdirSync(gitDir)
  fs.writeFileSync(path.join(gitDir, 'HEAD'), 'ref: refs/heads/main\n')

  const info = resolveWorktree(root)
  assert.notEqual(info, null)
  assert.equal(info.repoRoot, root)
  assert.equal(info.worktreeRoot, root)
  assert.equal(info.isMainWorktree, true)
  assert.equal(info.branch, 'main')
})

test('resolveWorktree reads branch from a nested subdirectory', (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  fs.mkdirSync(path.join(root, '.git'))
  fs.writeFileSync(path.join(root, '.git', 'HEAD'), 'ref: refs/heads/feat/thing\n')
  const sub = path.join(root, 'src', 'lib')
  fs.mkdirSync(sub, { recursive: true })

  const info = resolveWorktree(sub)
  assert.notEqual(info, null)
  assert.equal(info.repoRoot, root)
  assert.equal(info.branch, 'feat/thing')
})

test('resolveWorktree resolves from a file path (not just directories)', (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  fs.mkdirSync(path.join(root, '.git'))
  fs.writeFileSync(path.join(root, '.git', 'HEAD'), 'ref: refs/heads/main\n')
  const filePath = path.join(root, 'index.ts')
  fs.writeFileSync(filePath, '')

  const info = resolveWorktree(filePath)
  assert.notEqual(info, null)
  assert.equal(info.repoRoot, root)
})

test('resolveWorktree reports a detached HEAD as a short sha', (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  fs.mkdirSync(path.join(root, '.git'))
  fs.writeFileSync(
    path.join(root, '.git', 'HEAD'),
    'a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\n'
  )

  const info = resolveWorktree(root)
  assert.notEqual(info, null)
  assert.equal(info.branch, 'a1b2c3d4')
})

test('resolveWorktree identifies a linked worktree (.git file)', (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  // Set up the main repo's .git directory
  const mainGitDir = path.join(root, 'main-repo', '.git')
  fs.mkdirSync(mainGitDir, { recursive: true })
  fs.writeFileSync(path.join(mainGitDir, 'HEAD'), 'ref: refs/heads/main\n')

  // Set up the worktree admin dir inside the main repo's .git/worktrees/
  const adminDir = path.join(mainGitDir, 'worktrees', 'my-wt')
  fs.mkdirSync(adminDir, { recursive: true })
  fs.writeFileSync(path.join(adminDir, 'HEAD'), 'ref: refs/heads/feature-x\n')
  fs.writeFileSync(path.join(adminDir, 'commondir'), '../../\n')

  // Set up the linked worktree directory with a .git file
  const linkedDir = path.join(root, 'linked-wt')
  fs.mkdirSync(linkedDir, { recursive: true })
  fs.writeFileSync(path.join(linkedDir, '.git'), `gitdir: ${adminDir}\n`)

  const info = resolveWorktree(linkedDir)
  assert.notEqual(info, null)
  assert.equal(info.repoRoot, path.join(root, 'main-repo'))
  assert.equal(info.worktreeRoot, linkedDir)
  assert.equal(info.isMainWorktree, false)
  assert.equal(info.branch, 'feature-x')
})

test('resolveWorktree returns null for invalid input', () => {
  assert.equal(resolveWorktree(''), null)
  assert.equal(resolveWorktree(null), null)
  assert.equal(resolveWorktree('\\\\?\\C:\\secret'), null)
})

// ── worktreesForIpc ────────────────────────────────────────────────────

test('worktreesForIpc resolves multiple cwds and deduplicates', async (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  fs.mkdirSync(path.join(root, '.git'))
  fs.writeFileSync(path.join(root, '.git', 'HEAD'), 'ref: refs/heads/main\n')

  const result = await worktreesForIpc([root, root])
  const keys = Object.keys(result)
  assert.equal(keys.length, 1, 'duplicate cwd produces one entry')
  assert.equal(result[root].repoRoot, root)
})

test('worktreesForIpc skips non-string and empty entries', async () => {
  const result = await worktreesForIpc([null, '', '   ', 42, undefined])
  assert.deepEqual(result, {})
})

test('worktreesForIpc returns null for paths outside a git repo', async (t) => {
  const root = mkTmpDir()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const result = await worktreesForIpc([root])
  assert.equal(result[root], null)
})
