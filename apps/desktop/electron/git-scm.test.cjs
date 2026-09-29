'use strict'

const assert = require('node:assert/strict')
const { execFileSync } = require('node:child_process')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  parseStatus,
  statusWord,
  gitStatusForIpc,
  gitDiffForIpc,
  gitStageForIpc,
  gitUnstageForIpc,
  gitCommitForIpc,
  gitBranchesForIpc,
  gitCheckoutForIpc,
  gitCreateBranchForIpc,
  gitDeleteBranchForIpc,
  gitLogForIpc,
  gitCommitDiffForIpc,
  gitStashListForIpc,
  gitStashPushForIpc,
  gitStashActionForIpc,
  gitApplyHunkForIpc,
  gitRevertEditForIpc
} = require('./git-scm.cjs')

// ── Pure parser tests (no git needed) ───────────────────────────────────────

test('statusWord maps porcelain codes to friendly words', () => {
  assert.equal(statusWord('M', '.'), 'modified')
  assert.equal(statusWord('A', '.'), 'added')
  assert.equal(statusWord('D', '.'), 'deleted')
  assert.equal(statusWord('R', '.'), 'renamed')
  assert.equal(statusWord('.', 'M'), 'modified')
  assert.equal(statusWord('U', 'U'), 'conflicted')
})

test('parseStatus reads branch header + ahead/behind', () => {
  const z =
    '# branch.head main\u0000' +
    '# branch.upstream origin/main\u0000' +
    '# branch.ab +2 -1\u0000'
  const r = parseStatus(z)
  assert.equal(r.branch, 'main')
  assert.equal(r.upstream, 'origin/main')
  assert.equal(r.ahead, 2)
  assert.equal(r.behind, 1)
})

test('parseStatus splits staged vs unstaged vs untracked', () => {
  const z =
    '# branch.head main\u0000' +
    // staged modification (X=M, Y=.)
    '1 M. N... 100644 100644 100644 aaa bbb staged.txt\u0000' +
    // unstaged modification (X=., Y=M)
    '1 .M N... 100644 100644 100644 ccc ccc work.txt\u0000' +
    // both staged and unstaged (X=M, Y=M)
    '1 MM N... 100644 100644 100644 ddd eee both.txt\u0000' +
    // untracked
    '? newfile.txt\u0000'
  const r = parseStatus(z)

  assert.deepEqual(
    r.staged.map(e => e.path).sort(),
    ['both.txt', 'staged.txt']
  )
  assert.deepEqual(
    r.unstaged.map(e => e.path).sort(),
    ['both.txt', 'work.txt']
  )
  assert.deepEqual(r.untracked.map(e => e.path), ['newfile.txt'])
})

test('parseStatus handles rename records (kind 2 + origPath)', () => {
  const z =
    '# branch.head main\u0000' +
    '2 R. N... 100644 100644 100644 aaa bbb R100 new-name.txt\u0000old-name.txt\u0000'
  const r = parseStatus(z)
  assert.equal(r.staged.length, 1)
  assert.equal(r.staged[0].path, 'new-name.txt')
  assert.equal(r.staged[0].origPath, 'old-name.txt')
  assert.equal(r.staged[0].status, 'renamed')
})

test('parseStatus records conflicts under conflicted', () => {
  const z =
    '# branch.head main\u0000' +
    'u UU N... 100644 100644 100644 100644 hash1 hash2 hash3 both-edited.txt\u0000'
  const r = parseStatus(z)
  assert.equal(r.conflicted.length, 1)
  assert.equal(r.conflicted[0].path, 'both-edited.txt')
})

test('parseStatus keeps filenames with spaces intact', () => {
  const z = '# branch.head main\u0000' + '1 .M N... 100644 100644 100644 ccc ccc my file name.txt\u0000'
  const r = parseStatus(z)
  assert.equal(r.unstaged[0].path, 'my file name.txt')
})

// ── End-to-end against a real temp git repo ─────────────────────────────────

function gitAvailable() {
  try {
    execFileSync('git', ['--version'], { stdio: 'ignore' })
    return true
  } catch {
    return false
  }
}

function initRepo() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-git-scm-'))
  const run = args => execFileSync('git', ['-C', root, ...args], { stdio: 'pipe' })
  run(['init', '-q'])
  run(['config', 'user.email', 'test@example.com'])
  run(['config', 'user.name', 'Test'])
  run(['config', 'commit.gpgsign', 'false'])
  run(['checkout', '-q', '-b', 'main'])
  return { root, run }
}

test('e2e: status → stage → commit → status round-trip', { skip: !gitAvailable() }, async t => {
  const { root, run } = initRepo()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  // Seed an initial commit so HEAD exists.
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\n')
  run(['add', 'README.md'])
  run(['commit', '-q', '-m', 'init'])

  // Create one untracked + modify the tracked file.
  fs.writeFileSync(path.join(root, 'new.txt'), 'hello\n')
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\nmore\n')

  let st = await gitStatusForIpc('git', root)
  assert.equal(st.ok, true)
  assert.equal(st.branch, 'main')
  assert.equal(st.untracked.map(e => e.path).includes('new.txt'), true)
  assert.equal(st.unstaged.map(e => e.path).includes('README.md'), true)

  // Stage both, confirm they move to staged.
  const staged = await gitStageForIpc('git', root, ['new.txt', 'README.md'])
  assert.equal(staged.ok, true)
  st = await gitStatusForIpc('git', root)
  assert.equal(st.staged.map(e => e.path).sort().join(','), 'README.md,new.txt')
  assert.equal(st.untracked.length, 0)

  // Unstage new.txt → back to untracked.
  const unstaged = await gitUnstageForIpc('git', root, ['new.txt'])
  assert.equal(unstaged.ok, true)
  st = await gitStatusForIpc('git', root)
  assert.equal(st.untracked.map(e => e.path).includes('new.txt'), true)

  // Commit the staged README change.
  const committed = await gitCommitForIpc('git', root, 'update readme')
  assert.equal(committed.ok, true)
  st = await gitStatusForIpc('git', root)
  assert.equal(st.staged.length, 0)
})

test('e2e: diff returns a unified patch for a worktree change', { skip: !gitAvailable() }, async t => {
  const { root, run } = initRepo()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  fs.writeFileSync(path.join(root, 'a.txt'), 'one\ntwo\n')
  run(['add', 'a.txt'])
  run(['commit', '-q', '-m', 'init'])
  fs.writeFileSync(path.join(root, 'a.txt'), 'one\nTWO\nthree\n')

  const res = await gitDiffForIpc('git', root, 'a.txt', false)
  assert.equal(res.ok, true)
  assert.match(res.diff, /\+three/)
  assert.match(res.diff, /-two/)
})

test('e2e: empty commit message is rejected', { skip: !gitAvailable() }, async t => {
  const { root } = initRepo()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))

  const res = await gitCommitForIpc('git', root, '   ')
  assert.equal(res.ok, false)
  assert.equal(res.error, 'empty-message')
})

test('e2e: status on a non-repo returns not-a-repo', { skip: !gitAvailable() }, async t => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-notrepo-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))

  const res = await gitStatusForIpc('git', dir)
  assert.equal(res.ok, false)
  assert.equal(res.error, 'not-a-repo')
})

// ── Advanced ops: branches / log / stash / hunk staging ─────────────────────

function seededRepo(t) {
  const { root, run } = initRepo()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\nline2\nline3\n')
  run(['add', 'README.md'])
  run(['commit', '-q', '-m', 'init'])
  return { root, run }
}

test('e2e: branches lists current + created branches', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)

  const created = await gitCreateBranchForIpc('git', root, 'feature/x')
  assert.equal(created.ok, true)

  const res = await gitBranchesForIpc('git', root)
  assert.equal(res.ok, true)
  assert.equal(res.current, 'feature/x')
  assert.equal(
    res.local.map(b => b.name).sort().join(','),
    'feature/x,main'
  )
  assert.equal(res.local.find(b => b.name === 'feature/x').current, true)
})

test('e2e: checkout switches branches', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  await gitCreateBranchForIpc('git', root, 'dev')

  const back = await gitCheckoutForIpc('git', root, 'main')
  assert.equal(back.ok, true)

  const res = await gitBranchesForIpc('git', root)
  assert.equal(res.current, 'main')
})

test('e2e: delete branch removes it', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  await gitCreateBranchForIpc('git', root, 'temp')
  await gitCheckoutForIpc('git', root, 'main')

  const del = await gitDeleteBranchForIpc('git', root, 'temp', { force: true })
  assert.equal(del.ok, true)

  const res = await gitBranchesForIpc('git', root)
  assert.equal(res.local.find(b => b.name === 'temp'), undefined)
})

test('e2e: log returns commits with metadata', { skip: !gitAvailable() }, async t => {
  const { root, run } = seededRepo(t)
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\nchanged\n')
  run(['commit', '-q', '-am', 'second commit'])

  const res = await gitLogForIpc('git', root, { limit: 10 })
  assert.equal(res.ok, true)
  assert.equal(res.commits.length, 2)
  assert.equal(res.commits[0].subject, 'second commit')
  assert.match(res.commits[0].sha, /^[0-9a-f]{40}$/)
  assert.equal(res.commits[0].author, 'Test')
})

test('e2e: commit diff returns the patch for a sha', { skip: !gitAvailable() }, async t => {
  const { root, run } = seededRepo(t)
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\nline2 edited\nline3\n')
  run(['commit', '-q', '-am', 'edit line2'])
  const sha = execFileSync('git', ['-C', root, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()

  const res = await gitCommitDiffForIpc('git', root, sha)
  assert.equal(res.ok, true)
  assert.match(res.diff, /edit line2/)
  assert.match(res.diff, /\+line2 edited/)
})

test('e2e: commit diff rejects a bad sha', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const res = await gitCommitDiffForIpc('git', root, 'not a sha!!')
  assert.equal(res.ok, false)
  assert.equal(res.error, 'bad-sha')
})

test('e2e: stash push → list → pop round-trip', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\nWIP change\n')

  const pushed = await gitStashPushForIpc('git', root, { message: 'my wip' })
  assert.equal(pushed.ok, true)

  const list = await gitStashListForIpc('git', root)
  assert.equal(list.ok, true)
  assert.equal(list.stashes.length, 1)
  assert.match(list.stashes[0].ref, /^stash@\{0\}$/)
  assert.match(list.stashes[0].subject, /my wip/)

  // Working tree should be clean after the stash.
  let st = await gitStatusForIpc('git', root)
  assert.equal(st.unstaged.length, 0)

  const popped = await gitStashActionForIpc('git', root, 'pop', 'stash@{0}')
  assert.equal(popped.ok, true)

  st = await gitStatusForIpc('git', root)
  assert.equal(st.unstaged.map(e => e.path).includes('README.md'), true)
})

test('e2e: stash action rejects a malformed ref', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  // Bad ref is simply ignored (not appended); drop with no stashes fails in git,
  // but the point is the malformed ref never reaches the arg line.
  const res = await gitStashActionForIpc('git', root, 'bogus', 'stash@{0}')
  assert.equal(res.ok, false)
  assert.equal(res.error, 'bad-action')
})

test('e2e: apply hunk stages a single hunk via patch', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  // Modify the file, grab the unstaged diff, then stage exactly that patch.
  fs.writeFileSync(path.join(root, 'README.md'), '# repo\nline2 CHANGED\nline3\n')

  const diff = await gitDiffForIpc('git', root, 'README.md', false)
  assert.equal(diff.ok, true)
  assert.match(diff.diff, /@@/)

  const applied = await gitApplyHunkForIpc('git', root, diff.diff, { reverse: false })
  assert.equal(applied.ok, true)

  // After applying to the index, the change shows up as staged.
  const st = await gitStatusForIpc('git', root)
  assert.equal(st.staged.map(e => e.path).includes('README.md'), true)
})

test('e2e: apply hunk rejects an empty patch', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const res = await gitApplyHunkForIpc('git', root, '   ')
  assert.equal(res.ok, false)
  assert.equal(res.error, 'empty-patch')
})

// ── Edit review: revert (reject) ────────────────────────────────────────────

test('e2e: revertEdit reverse-applies a diff to the worktree', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const file = path.join(root, 'README.md')
  const original = fs.readFileSync(file, 'utf8')

  // Simulate an agent edit, capture its diff, then reject it.
  fs.writeFileSync(file, '# repo\nAGENT EDIT\nline3\n')
  const diff = await gitDiffForIpc('git', root, 'README.md', false)
  assert.equal(diff.ok, true)

  const res = await gitRevertEditForIpc('git', root, { path: 'README.md', diff: diff.diff, isNew: false })
  assert.equal(res.ok, true)
  // File restored to its pre-edit content.
  assert.equal(fs.readFileSync(file, 'utf8'), original)
})

test('e2e: revertEdit deletes a newly-created file', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const newFile = path.join(root, 'brand-new.txt')
  fs.writeFileSync(newFile, 'created by agent\n')

  const res = await gitRevertEditForIpc('git', root, { path: 'brand-new.txt', isNew: true })
  assert.equal(res.ok, true)
  assert.equal(fs.existsSync(newFile), false)
})

test('e2e: revertEdit preserves unrelated edits elsewhere in the file', { skip: !gitAvailable() }, async t => {
  // A larger file so the agent's hunk and the user's hunk don't share context.
  const { root, run } = initRepo()
  t.after(() => fs.rmSync(root, { recursive: true, force: true }))
  const file = path.join(root, 'big.txt')
  const lines = Array.from({ length: 20 }, (_, i) => `line ${i + 1}`)
  fs.writeFileSync(file, lines.join('\n') + '\n')
  run(['add', 'big.txt'])
  run(['commit', '-q', '-m', 'init big'])

  // Agent edits line 3; capture that diff.
  const agentLines = [...lines]
  agentLines[2] = 'AGENT EDIT'
  fs.writeFileSync(file, agentLines.join('\n') + '\n')
  const diff = await gitDiffForIpc('git', root, 'big.txt', false)
  assert.equal(diff.ok, true)

  // A separate edit lands on line 18 — far from the agent's hunk.
  const bothLines = [...agentLines]
  bothLines[17] = 'USER EDIT'
  fs.writeFileSync(file, bothLines.join('\n') + '\n')

  // Rejecting the agent's diff undoes line 3, keeps line 18.
  const res = await gitRevertEditForIpc('git', root, { path: 'big.txt', diff: diff.diff, isNew: false })
  assert.equal(res.ok, true)
  const after = fs.readFileSync(file, 'utf8')
  assert.match(after, /USER EDIT/)
  assert.doesNotMatch(after, /AGENT EDIT/)
  assert.match(after, /line 3/) // restored
})

test('e2e: revertEdit fails safely when the diff no longer applies', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const file = path.join(root, 'README.md')

  fs.writeFileSync(file, '# repo\nAGENT EDIT\nline3\n')
  const diff = await gitDiffForIpc('git', root, 'README.md', false)

  // The file is then rewritten entirely, so the agent's diff context is gone.
  fs.writeFileSync(file, 'completely different content\n')

  const res = await gitRevertEditForIpc('git', root, { path: 'README.md', diff: diff.diff, isNew: false })
  // Must not corrupt the file — reject cleanly instead.
  assert.equal(res.ok, false)
  assert.equal(fs.readFileSync(file, 'utf8'), 'completely different content\n')
})


test('e2e: revertEdit rejects a bad path', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const res = await gitRevertEditForIpc('git', root, { path: '', diff: 'x' })
  assert.equal(res.ok, false)
  assert.equal(res.error, 'bad-path')
})



// ── Edit review: path containment for new-file deletion ─────────────────────

function outsideVictim(t, root) {
  const dir = fs.mkdtempSync(path.join(path.dirname(root), 'hermes-victim-'))
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }))
  const file = path.join(dir, 'victim.txt')
  fs.writeFileSync(file, 'must survive\n')
  return { dir, file }
}

test('e2e: revertEdit refuses ../ traversal out of the repo', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const victim = outsideVictim(t, root)
  const rel = path.relative(root, victim.file)
  assert.ok(rel.startsWith('..'))

  const res = await gitRevertEditForIpc('git', root, { path: rel, isNew: true })
  assert.equal(res.ok, false)
  assert.equal(res.error, 'bad-path')
  assert.equal(fs.readFileSync(victim.file, 'utf8'), 'must survive\n')

  const sneaky = await gitRevertEditForIpc('git', root, { path: `sub/../../${rel}`, isNew: true })
  assert.equal(sneaky.ok, false)
  assert.equal(fs.existsSync(victim.file), true)
})

test('e2e: revertEdit refuses absolute paths', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const victim = outsideVictim(t, root)

  const res = await gitRevertEditForIpc('git', root, { path: victim.file, isNew: true })
  assert.equal(res.ok, false)
  assert.equal(res.error, 'bad-path')
  assert.equal(fs.existsSync(victim.file), true)
})

test('e2e: revertEdit refuses the repo root and directories', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  fs.mkdirSync(path.join(root, 'dir'))
  fs.writeFileSync(path.join(root, 'dir', 'keep.txt'), 'x\n')

  for (const rel of ['.', 'dir', 'dir/..']) {
    const res = await gitRevertEditForIpc('git', root, { path: rel, isNew: true })
    assert.equal(res.ok, false, rel)
    assert.equal(res.error, 'bad-path', rel)
  }
  assert.equal(fs.existsSync(path.join(root, 'dir', 'keep.txt')), true)
  assert.equal(fs.existsSync(path.join(root, 'README.md')), true)
})

test('e2e: revertEdit refuses a symlinked parent that escapes the repo', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  const victim = outsideVictim(t, root)
  try {
    fs.symlinkSync(victim.dir, path.join(root, 'escape'), 'junction')
  } catch {
    t.skip('cannot create directory symlinks here')
    return
  }

  const res = await gitRevertEditForIpc('git', root, { path: 'escape/victim.txt', isNew: true })
  assert.equal(res.ok, false)
  assert.equal(res.error, 'bad-path')
  assert.equal(fs.existsSync(victim.file), true)
})

test('e2e: revertEdit still deletes a nested new file inside the repo', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  fs.mkdirSync(path.join(root, 'src', 'deep'), { recursive: true })
  const newFile = path.join(root, 'src', 'deep', 'made.txt')
  fs.writeFileSync(newFile, 'agent\n')

  const res = await gitRevertEditForIpc('git', root, { path: 'src/deep/made.txt', isNew: true })
  assert.equal(res.ok, true)
  assert.equal(fs.existsSync(newFile), false)

  const gone = await gitRevertEditForIpc('git', root, { path: 'src/deep/made.txt', isNew: true })
  assert.equal(gone.ok, true)
})

test('e2e: stash action rejects a malformed ref instead of using the latest stash', { skip: !gitAvailable() }, async t => {
  const { root } = seededRepo(t)
  fs.writeFileSync(path.join(root, 'README.md'), 'stashed change\n')
  assert.equal((await gitStashPushForIpc('git', root, {})).ok, true)

  for (const ref of ['stash@{x}', 'HEAD', '--all', '']) {
    const res = await gitStashActionForIpc('git', root, 'drop', ref)
    assert.equal(res.ok, false, ref)
    assert.equal(res.error, 'bad-stash-ref', ref)
  }
  const list = await gitStashListForIpc('git', root)
  assert.equal(list.stashes.length, 1)
})
