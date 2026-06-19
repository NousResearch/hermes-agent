// Hardening tests for the chat file-path "open in default app" / "reveal"
// handlers (see main.cjs hermes:openPath / hermes:revealInFolder). These cover
// resolveOpenablePathForIpc + assertOpenableInDefaultApp from hardening.cjs;
// they run under plain `node --test` (no Electron) because hardening.cjs only
// requires node builtins.

const assert = require('node:assert/strict')
const fs = require('node:fs')
const os = require('node:os')
const path = require('node:path')
const test = require('node:test')

const {
  OPENABLE_DEFAULT_APP_EXTENSIONS,
  assertOpenableInDefaultApp,
  rejectRemotePathSyntax,
  resolveOpenablePathForIpc
} = require('./hardening.cjs')

function tmpDir(prefix) {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix))
}

test('opens an allowlisted file confined to the workspace', async () => {
  const dir = tmpDir('hermes-openpath-')
  const file = path.join(dir, 'report.pdf')
  fs.writeFileSync(file, '%PDF-1.4')

  const { realPath } = await resolveOpenablePathForIpc(file, {
    baseDir: dir,
    purpose: 'Open file',
    requireWithinBase: true
  })

  assert.equal(realPath, fs.realpathSync(file))
  assert.doesNotThrow(() => assertOpenableInDefaultApp(realPath))
})

test('blocks executable / script / launcher / web extensions from open-in-default-app', () => {
  for (const name of ['evil.command', 'evil.sh', 'evil.app', 'evil.desktop', 'evil.bat', 'evil.html', 'evil.svg']) {
    assert.throws(() => assertOpenableInDefaultApp(`/ws/${name}`), /not-openable|cannot be opened/i)
  }
})

test('the open allowlist excludes html and svg but includes inert media/docs', () => {
  assert.ok(!OPENABLE_DEFAULT_APP_EXTENSIONS.has('.html'))
  assert.ok(!OPENABLE_DEFAULT_APP_EXTENSIONS.has('.svg'))
  assert.ok(OPENABLE_DEFAULT_APP_EXTENSIONS.has('.png'))
  assert.ok(OPENABLE_DEFAULT_APP_EXTENSIONS.has('.pdf'))
})

test('rejects a path outside the workspace when confinement is required', async () => {
  const dir = tmpDir('hermes-openpath-')
  const outside = tmpDir('hermes-outside-')
  const file = path.join(outside, 'note.txt')
  fs.writeFileSync(file, 'x')

  await assert.rejects(
    () => resolveOpenablePathForIpc(file, { baseDir: dir, purpose: 'Open file', requireWithinBase: true }),
    /outside the current workspace|outside-workspace/i
  )
})

test('blocks sensitive files (.env) even inside the workspace', async () => {
  const dir = tmpDir('hermes-openpath-')
  const env = path.join(dir, '.env')
  fs.writeFileSync(env, 'SECRET=1')

  await assert.rejects(
    () => resolveOpenablePathForIpc(env, { baseDir: dir, purpose: 'Open file', requireWithinBase: true }),
    /sensitive/i
  )
})

test('follows symlinks and re-checks the real path (workspace escape blocked)', async () => {
  const dir = tmpDir('hermes-openpath-')
  const outside = tmpDir('hermes-outside-')
  const realFile = path.join(outside, 'secret.pdf')
  fs.writeFileSync(realFile, '%PDF')
  const link = path.join(dir, 'innocent.pdf')

  try {
    fs.symlinkSync(realFile, link)
  } catch {
    return // symlink creation may be unavailable (unprivileged CI); skip.
  }

  await assert.rejects(
    () => resolveOpenablePathForIpc(link, { baseDir: dir, purpose: 'Open file', requireWithinBase: true }),
    /outside the current workspace|outside-workspace/i
  )
})

test('rejects network / UNC paths', () => {
  assert.throws(() => rejectRemotePathSyntax('\\\\server\\share\\x.txt'), /network|unc|remote/i)
  assert.throws(() => rejectRemotePathSyntax('//server/share/x.txt'), /network|unc|remote/i)
})

test('reveal allows a file outside the workspace (requireWithinBase: false)', async () => {
  const dir = tmpDir('hermes-reveal-')
  const file = path.join(dir, 'note.txt')
  fs.writeFileSync(file, 'hi')

  const { realPath } = await resolveOpenablePathForIpc(file, { purpose: 'Reveal file', requireWithinBase: false })

  assert.equal(realPath, fs.realpathSync(file))
})

test('rejects paths containing control characters (open-dialog spoofing guard)', async () => {
  const dir = tmpDir('hermes-openpath-')
  let file

  try {
    file = path.join(dir, 'evil\nreport.pdf')
    fs.writeFileSync(file, '%PDF')
  } catch {
    return // some filesystems disallow newline filenames; skip.
  }

  await assert.rejects(
    () => resolveOpenablePathForIpc(file, { baseDir: dir, purpose: 'Open file', requireWithinBase: true }),
    /control character/i
  )
})

test('rejects a directory and a missing file', async () => {
  const dir = tmpDir('hermes-openpath-')

  await assert.rejects(
    () => resolveOpenablePathForIpc(dir, { baseDir: dir, purpose: 'Open file', requireWithinBase: true }),
    /directory/i
  )
  await assert.rejects(
    () => resolveOpenablePathForIpc(path.join(dir, 'nope.pdf'), { baseDir: dir, purpose: 'Open file' }),
    /does not exist/i
  )
})
