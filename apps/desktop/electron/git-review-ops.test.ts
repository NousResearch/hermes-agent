import assert from 'node:assert/strict'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, test } from 'vitest'

import {
  fileDiffVsHead,
  gitFor,
  repoStatus,
  resolveRenamePath,
  REVIEW_FILE_CAP,
  reviewDiff,
  reviewList,
  reviewRevert,
  reviewStage,
  reviewUnstage
} from './git-review-ops'

const tempDirs: string[] = []

afterEach(() => {
  for (const dir of tempDirs.splice(0)) {
    fs.rmSync(dir, { force: true, recursive: true })
  }
})

function makeRepo() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-git-status-'))

  tempDirs.push(dir)
  execFileSync('git', ['init', '-q'], { cwd: dir })
  execFileSync('git', ['config', 'user.email', 'hermes-test@example.com'], { cwd: dir })
  execFileSync('git', ['config', 'user.name', 'Hermes Test'], { cwd: dir })
  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'tracked\n')
  execFileSync('git', ['add', 'tracked.txt'], { cwd: dir })
  execFileSync('git', ['commit', '-qm', 'initial'], { cwd: dir })

  return dir
}

test('resolveRenamePath: plain path is unchanged', () => {
  assert.equal(resolveRenamePath('src/a.ts'), 'src/a.ts')
})

test('gitFor accepts an internally resolved git binary path containing spaces', () => {
  assert.doesNotThrow(() => gitFor(process.cwd(), 'C:\\Program Files\\Git\\cmd\\git.exe'))
})

test('gitFor runs git through a spaced binary path', async () => {
  if (process.platform !== 'win32') {
    return
  }

  const gitBin = path.join(process.env.ProgramFiles || String.raw`C:\Program Files`, 'Git', 'cmd', 'git.exe')

  if (!fs.existsSync(gitBin)) {
    return
  }

  const repo = makeRepo()

  fs.writeFileSync(path.join(repo, 'changed.txt'), 'review me\n')

  const status = await gitFor(repo, gitBin).status()

  assert.equal(status.not_added.includes('changed.txt'), true)
})

test('resolveRenamePath: simple rename resolves to the new path', () => {
  assert.equal(resolveRenamePath('old.ts => new.ts'), 'new.ts')
})

test('resolveRenamePath: brace rename resolves to the new path', () => {
  assert.equal(resolveRenamePath('src/{old => new}/file.ts'), 'src/new/file.ts')
})

test('resolveRenamePath: brace rename collapsing a segment', () => {
  assert.equal(resolveRenamePath('src/{lib => }/file.ts'), 'src/file.ts')
})

test('repoStatus reports an untracked directory without recursively listing its contents', async () => {
  const dir = makeRepo()
  const nested = path.join(dir, 'generated', 'deep')

  fs.mkdirSync(nested, { recursive: true })
  fs.writeFileSync(path.join(nested, 'large-output.txt'), 'generated\n')

  const status = await repoStatus(dir, 'git')

  assert.ok(status)
  assert.equal(status.untracked, 1)
  assert.equal(status.changed, 1)
  assert.deepEqual(
    status.files.map(file => file.path),
    ['generated/']
  )
})

test('reviewList reports an untracked directory without recursively listing its contents', async () => {
  const dir = makeRepo()
  const nested = path.join(dir, 'browser-profile', 'Default', 'Cache')

  fs.mkdirSync(nested, { recursive: true })

  for (let i = 0; i < 20; i++) {
    fs.writeFileSync(path.join(nested, `cache-${i}.bin`), 'generated\n')
  }

  const result = await reviewList(dir, 'uncommitted', null, 'git')

  assert.deepEqual(
    result.files.map(file => file.path),
    ['browser-profile/']
  )
})

test('reviewList caps the file payload returned to the renderer', async () => {
  const dir = makeRepo()

  for (let i = 0; i < REVIEW_FILE_CAP + 10; i++) {
    fs.writeFileSync(path.join(dir, `untracked-${String(i).padStart(4, '0')}.txt`), 'generated\n')
  }

  const result = await reviewList(dir, 'uncommitted', null, 'git')

  assert.equal(result.files.length, REVIEW_FILE_CAP)
})

// ── reviewDiff: the click → bottom-panel payload ─────────────────────────────
// `reviewList` collapses an untracked directory into one `dir/` row (asserted
// above). Clicking that row used to hand `dir/` to `git diff --no-index --
// /dev/null dir/`, which pairs the operands as trees, fails looking for
// `dir/null`, and prints nothing — so the pane rendered "No diff to show"
// under a fully populated header.

const uncommittedDiff = (dir: string, filePath: string, staged = false) =>
  reviewDiff(dir, filePath, 'uncommitted', null, staged, 'git')

test('reviewDiff synthesizes an all-add diff for an untracked file', async () => {
  const dir = makeRepo()

  fs.writeFileSync(path.join(dir, 'fresh.txt'), 'alpha\nbeta\n')

  const diff = await uncommittedDiff(dir, 'fresh.txt')

  assert.match(diff, /\+alpha/)
  assert.match(diff, /\+beta/)
})

test('reviewDiff expands an untracked directory into its files', async () => {
  const dir = makeRepo()

  fs.mkdirSync(path.join(dir, 'newdir', 'sub'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'newdir', 'one.txt'), 'first\n')
  fs.writeFileSync(path.join(dir, 'newdir', 'sub', 'two.txt'), 'second\n')

  const diff = await uncommittedDiff(dir, 'newdir/')

  assert.notEqual(diff.trim(), '')
  assert.match(diff, /\+first/)
  assert.match(diff, /\+second/)
  // Every file is named, so the renderer can label a multi-file payload.
  assert.match(diff, /newdir\/one\.txt/)
  assert.match(diff, /newdir\/sub\/two\.txt/)
})

test('reviewDiff expands an untracked directory whose path contains spaces', async () => {
  const dir = makeRepo()

  fs.mkdirSync(path.join(dir, 'Fallout Vault', '20 Projects'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'Fallout Vault', '20 Projects', 'note.md'), 'vault note\n')

  const diff = await uncommittedDiff(dir, 'Fallout Vault/')

  assert.match(diff, /\+vault note/)
})

test('reviewDiff skips gitignored files when expanding an untracked directory', async () => {
  const dir = makeRepo()

  fs.writeFileSync(path.join(dir, '.gitignore'), '*.log\n')
  fs.mkdirSync(path.join(dir, 'logs'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'logs', 'keep.txt'), 'kept\n')
  fs.writeFileSync(path.join(dir, 'logs', 'noisy.log'), 'ignored\n')

  const diff = await uncommittedDiff(dir, 'logs/')

  assert.match(diff, /\+kept/)
  assert.doesNotMatch(diff, /\+ignored/)
})

test('reviewDiff caps a huge untracked directory and says what it dropped', async () => {
  const dir = makeRepo()

  fs.mkdirSync(path.join(dir, 'generated'), { recursive: true })

  for (let i = 0; i < 60; i++) {
    fs.writeFileSync(path.join(dir, 'generated', `f-${String(i).padStart(3, '0')}.txt`), `line ${i}\n`)
  }

  const diff = await uncommittedDiff(dir, 'generated/')

  assert.match(diff, /\+line 0/)
  // Truncation is never silent.
  assert.match(diff, /10 more file\(s\) omitted/)
})

test('reviewDiff returns empty for a nested git repo the outer repo cannot see into', async () => {
  const dir = makeRepo()
  const nested = path.join(dir, 'nested_repo')

  fs.mkdirSync(nested, { recursive: true })
  execFileSync('git', ['init', '-q'], { cwd: nested })
  fs.writeFileSync(path.join(nested, 'inner.txt'), 'inner\n')

  // Opaque to the outer repo — the pane shows the folder empty-state for this,
  // not the generic "No diff to show".
  assert.equal(await uncommittedDiff(dir, 'nested_repo/'), '')
})

test('reviewDiff shows staged AND unstaged changes for a partially staged file', async () => {
  const dir = makeRepo()
  const file = path.join(dir, 'tracked.txt')

  fs.writeFileSync(file, 'tracked\nstaged line\n')
  execFileSync('git', ['add', 'tracked.txt'], { cwd: dir })
  fs.writeFileSync(file, 'tracked\nstaged line\nunstaged line\n')

  // `staged: true` is what the row reports once anything is in the index; the
  // row's +/- counts sum both sides, so the diff has to as well.
  const diff = await uncommittedDiff(dir, 'tracked.txt', true)

  assert.match(diff, /\+staged line/)
  assert.match(diff, /\+unstaged line/)
})

test('reviewDiff falls back to the index diff when the repo has no commits yet', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-desktop-git-status-'))

  tempDirs.push(dir)
  execFileSync('git', ['init', '-q'], { cwd: dir })
  execFileSync('git', ['config', 'user.email', 'hermes-test@example.com'], { cwd: dir })
  execFileSync('git', ['config', 'user.name', 'Hermes Test'], { cwd: dir })
  fs.writeFileSync(path.join(dir, 'first.txt'), 'first commit pending\n')
  execFileSync('git', ['add', 'first.txt'], { cwd: dir })

  // No HEAD to diff against — the `--cached` fallback keeps the panel populated.
  assert.match(await uncommittedDiff(dir, 'first.txt', true), /\+first commit pending/)
})

test('fileDiffVsHead expands an untracked directory too', async () => {
  const dir = makeRepo()

  fs.mkdirSync(path.join(dir, 'preview-dir'), { recursive: true })
  fs.writeFileSync(path.join(dir, 'preview-dir', 'a.txt'), 'preview line\n')

  assert.match(await fileDiffVsHead(dir, 'preview-dir/', 'git'), /\+preview line/)
})

test('fileDiffVsHead still returns empty for a clean tracked file', async () => {
  const dir = makeRepo()

  assert.equal(await fileDiffVsHead(dir, 'tracked.txt', 'git'), '')
})

test('reviewDiff shows the real conflict body for an unmerged file', async () => {
  const dir = makeRepo()
  const file = path.join(dir, 'tracked.txt')
  const base = execFileSync('git', ['rev-parse', '--abbrev-ref', 'HEAD'], { cwd: dir }).toString().trim()

  execFileSync('git', ['checkout', '-q', '-b', 'other'], { cwd: dir })
  fs.writeFileSync(file, 'other side\n')
  execFileSync('git', ['commit', '-qam', 'other'], { cwd: dir })
  execFileSync('git', ['checkout', '-q', base], { cwd: dir })
  fs.writeFileSync(file, 'main side\n')
  execFileSync('git', ['commit', '-qam', 'main'], { cwd: dir })

  try {
    execFileSync('git', ['merge', 'other'], { cwd: dir, stdio: 'ignore' })
  } catch {
    // Conflicts, by design.
  }

  // An unmerged path reports `staged: true`. `--cached` answers that with the
  // bare "* Unmerged path" stub — no hunks, so the panel rendered one useless
  // line. HEAD..worktree carries the actual conflict.
  const diff = await uncommittedDiff(dir, 'tracked.txt', true)

  assert.match(diff, /<<<<<<< HEAD/)
  assert.match(diff, /other side/)
})

// A filename containing pathspec wildcards is matched as a GLOB by default, so
// `weird[1].txt` also selects `weird1.txt`. That leaked across every git call in
// this module: reads showed the wrong file's body, and the mutations changed the
// wrong file on disk. The decoy here is TRACKED and MODIFIED - the nastiest
// shape, with a real worktree diff to leak and real edits to destroy.
function makeGlobRepo() {
  const dir = makeRepo()

  fs.writeFileSync(path.join(dir, 'weird1.txt'), 'neighbour original\n')
  execFileSync('git', ['add', 'weird1.txt'], { cwd: dir })
  execFileSync('git', ['commit', '-qm', 'add neighbour'], { cwd: dir })
  fs.writeFileSync(path.join(dir, 'weird1.txt'), 'neighbour MODIFIED\n')
  fs.writeFileSync(path.join(dir, 'weird[1].txt'), 'clicked file\n')

  return dir
}

test('reviewDiff does not pull in neighbours of a file whose name looks like a glob', async () => {
  const dir = makeGlobRepo()

  // Without --literal-pathspecs the worktree probe `git diff -- weird[1].txt`
  // returns the TRACKED neighbour's diff, so the pane renders a file the user
  // never clicked.
  const diff = await uncommittedDiff(dir, 'weird[1].txt')

  assert.match(diff, /\+clicked file/)
  assert.doesNotMatch(diff, /neighbour MODIFIED/)
  assert.doesNotMatch(diff, /weird1\.txt/)
})

test('fileDiffVsHead does not pull in glob neighbours either', async () => {
  const dir = makeGlobRepo()
  const diff = await fileDiffVsHead(dir, 'weird[1].txt', 'git')

  assert.match(diff, /\+clicked file/)
  assert.doesNotMatch(diff, /neighbour MODIFIED/)
})

test('reviewStage stages only the selected file, not its glob neighbours', async () => {
  const dir = makeGlobRepo()

  await reviewStage(dir, 'weird[1].txt', 'git')

  const staged = execFileSync('git', ['diff', '--cached', '--name-only'], { cwd: dir }).toString()

  assert.match(staged, /weird\[1\]\.txt/)
  assert.doesNotMatch(staged, /^weird1\.txt$/m)
})

test('reviewUnstage unstages only the selected file, not its glob neighbours', async () => {
  const dir = makeGlobRepo()

  execFileSync('git', ['add', '-A'], { cwd: dir })
  await reviewUnstage(dir, 'weird[1].txt', 'git')

  const staged = execFileSync('git', ['diff', '--cached', '--name-only'], { cwd: dir }).toString()

  // The neighbour must stay staged; only the clicked file comes back out.
  assert.match(staged, /^weird1\.txt$/m)
  assert.doesNotMatch(staged, /weird\[1\]\.txt/)
})

test('reviewRevert does not discard edits to a glob neighbour', async () => {
  const dir = makeGlobRepo()

  // The destructive one: `git checkout HEAD -- 'weird[1].txt'` also restored
  // weird1.txt, silently throwing away the user's uncommitted edits.
  await reviewRevert(dir, 'weird[1].txt', 'git')

  assert.equal(fs.readFileSync(path.join(dir, 'weird1.txt'), 'utf8'), 'neighbour MODIFIED\n')
  assert.equal(fs.existsSync(path.join(dir, 'weird[1].txt')), false)
})

// The `filePath === null` variants take a different argv shape now that
// LITERAL_PATHSPECS leads every mutation ("stage all" / "unstage all" /
// "revert all" carry no pathspec at all, or the bare `.`).
test('reviewStage with no path stages everything', async () => {
  const dir = makeRepo()

  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'edited\n')
  fs.writeFileSync(path.join(dir, 'brand-new.txt'), 'new\n')
  await reviewStage(dir, null, 'git')

  const staged = execFileSync('git', ['diff', '--cached', '--name-only'], { cwd: dir }).toString()

  assert.match(staged, /tracked\.txt/)
  assert.match(staged, /brand-new\.txt/)
})

test('reviewUnstage with no path unstages everything', async () => {
  const dir = makeRepo()

  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'edited\n')
  execFileSync('git', ['add', '-A'], { cwd: dir })
  await reviewUnstage(dir, null, 'git')

  assert.equal(execFileSync('git', ['diff', '--cached', '--name-only'], { cwd: dir }).toString().trim(), '')
})

test('reviewRevert with no path restores tracked files and removes untracked ones', async () => {
  const dir = makeRepo()

  fs.writeFileSync(path.join(dir, 'tracked.txt'), 'edited\n')
  fs.writeFileSync(path.join(dir, 'brand-new.txt'), 'new\n')
  await reviewRevert(dir, null, 'git')

  // Checkout re-materializes the file through git's eol filters, so compare
  // content rather than bytes (core.autocrlf turns this into CRLF on Windows).
  assert.equal(fs.readFileSync(path.join(dir, 'tracked.txt'), 'utf8').replace(/\r\n/g, '\n'), 'tracked\n')
  assert.equal(fs.existsSync(path.join(dir, 'brand-new.txt')), false)
})
