import assert from 'node:assert/strict'
import type * as ChildProcess from 'node:child_process'
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import simpleGit from 'simple-git'
import { afterEach, test, vi } from 'vitest'

import {
  fileDiffVsHead,
  gitFor,
  repoStatus,
  resolveRenamePath,
  REVIEW_FILE_CAP,
  reviewCreatePr,
  reviewDiff,
  reviewList,
  reviewRevert,
  reviewStage,
  reviewUnstage,
  SIMPLE_GIT_UNSAFE_BINARY_WARN
} from './git-review-ops'
import type * as NoConsoleGit from './no-console-git'

// `runGh` shells to the `gh` CLI via execFile. Mock it so reviewCreatePr's gh
// invocation is controllable (real `gh` may be absent or slow in CI) while the
// repo setup below still uses the real execFileSync.
vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof ChildProcess>()

  return { ...actual, execFile: vi.fn(actual.execFile) }
})

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

test('gitFor accepts internally resolved git paths with restricted non-space characters', () => {
  // simple-git's whitelist is `/^([a-z]:)?([a-z0-9/.\_~-]+)$/i`, so parentheses
  // (`Program Files (x86)`), `+`, and accented profile dirs (`C:\Users\João\...`)
  // are rejected exactly like a space — a `/\s/` guess still throws on them.
  const restrictedBinaries = [
    String.raw`C:\Git(x86)\cmd\git.exe`,
    String.raw`C:\tools\git+portable\cmd\git.exe`,
    String.raw`C:\Users\João\AppData\Local\hermes\git\cmd\git.exe`
  ]

  for (const binary of restrictedBinaries) {
    assert.doesNotThrow(() => gitFor(process.cwd(), binary), `should accept ${binary}`)
  }
})

test('gitFor accepts a Windows no-console host tuple with restricted characters', async () => {
  vi.resetModules()
  vi.doMock('./no-console-git', async importOriginal => {
    const actual = await importOriginal<typeof NoConsoleGit>()

    return {
      ...actual,
      windowsGitHost: () => ({
        isWindows: true,
        pythonBin: String.raw`C:\Tools\python-3.14+build\python.exe`,
        scriptPath: String.raw`C:\Hermes\hermes-no-console-git.py`
      })
    }
  })

  try {
    const { gitFor: gitForWithHostTuple } = await import('./git-review-ops')

    assert.doesNotThrow(() => gitForWithHostTuple(process.cwd(), 'git'))
  } finally {
    vi.doUnmock('./no-console-git')
    vi.resetModules()
  }
})

test('gitFor suppresses only the known custom-binary warning and restores console.warn', () => {
  const spacedBin = String.raw`C:\Program Files\Git\cmd\git.exe`
  // `windowsGitHost()` resolves nothing in this process (no configured roots, no
  // HERMES_DESKTOP_PYTHON), so `gitBin` itself is what simple-git validates — the
  // spaced `Program Files` path, which warns once per factory call.
  const warnings: unknown[][] = []
  const originalWarn = console.warn

  const recordingWarn = (...args: unknown[]) => {
    warnings.push(args)
  }

  console.warn = recordingWarn

  try {
    for (let i = 0; i < 5; i += 1) {
      gitFor(process.cwd(), spacedBin)
    }

    assert.equal(console.warn, recordingWarn)

    // The escape hatch used directly still warns: the message gitFor filters is a
    // live emission of the installed simple-git, so the filter cannot go stale
    // silently (an upgrade that rewords it fails this test, not production).
    simpleGit({ baseDir: process.cwd(), binary: spacedBin, unsafe: { allowUnsafeCustomBinary: true } })
    console.warn('unrelated warning')
  } finally {
    console.warn = originalWarn
  }

  assert.deepEqual(warnings, [[SIMPLE_GIT_UNSAFE_BINARY_WARN], ['unrelated warning']])
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

const actualChildProcess = await vi.importActual<typeof ChildProcess>('node:child_process')

const mockExecFile = vi.mocked(await import('node:child_process')).execFile

type ExecFileCallback = (error: Error | null, stdout?: string, stderr?: string) => void

// `execFile` has overloaded declarations returning ChildProcess; the mock
// implementation only needs to drive its callback, so view it as a plain
// callable and set the implementation through the Mock typing.
function failGh(stderr: string): void {
  ;(
    mockExecFile as unknown as {
      mockImplementation: (
        impl: (file: string, args: string[], options: object, callback: ExecFileCallback) => void
      ) => unknown
    }
  ).mockImplementation((_bin: string, _args: string[], _opts: object, callback: ExecFileCallback) => {
    if (_bin !== 'gh') {
      actualChildProcess.execFile(_bin, _args, _opts, callback)

      return
    }

    const error = new Error('command failed')

    if (stderr) {
      ;(error as Error & { stderr?: string }).stderr = stderr
    }

    callback(error, '', stderr)
  })
}

test('reviewCreatePr surfaces gh stderr when pr create fails', async () => {
  const dir = makeRepo()

  failGh('no commits between main and feature')

  await assert.rejects(reviewCreatePr(dir, 'git', 'gh'), /no commits between main and feature/)
})

test('reviewCreatePr falls back to the generic message when gh reports no stderr', async () => {
  const dir = makeRepo()

  failGh('')

  await assert.rejects(reviewCreatePr(dir, 'git', 'gh'), /is gh installed and authenticated\?/)
})
