import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { execFileSync, spawnSync } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { JSDOM } from 'jsdom'
import { afterEach, expect, test, vi } from 'vitest'
import updateUi from '../tests/install/e2e-assets/update-ui.cjs'
import caseCollision from '../tests/install/e2e-assets/case-collision.cjs'
import sourceBranchProbe from '../tests/install/e2e-assets/source-branch-probe.cjs'

const windows = []
afterEach(() => {
  windows.splice(0).forEach(window => window.close())
  vi.useRealTimers()
})

function fixture({ details = true, available = true, statusOverride } = {}) {
  vi.useFakeTimers({ toFake: ['Date'] })
  vi.setSystemTime(0)
  const { window } = new JSDOM('<body></body>', { runScripts: 'outside-only' })
  windows.push(window)
  const { document } = window
  const clicks = []
  const addButton = (text, onClick) => {
    const button = document.createElement('button')
    button.textContent = text
    button.onclick = () => { clicks.push(text); onClick?.() }
    document.body.append(button)
    return button
  }
  const revealUpdate = () => addButton('Update now')
  const check = addButton('Check now', () => {
    check.disabled = true
    if (!available) return
    if (details) {
      const more = addButton("See what's new", () => { more.remove(); revealUpdate() })
    } else {
      revealUpdate()
    }
  })
  const status = statusOverride || { supported: true, behind: available ? 1 : 0 }
  window.hermesDesktop = { updates: { check: async () => status } }
  const page = {
    getByRole(role, { name }) {
      expect(role).toBe('button')
      const selected = () => [...document.querySelectorAll('button')].find(button => name.test(button.textContent))
      const locator = {
        first: () => locator,
        isVisible: async () => Boolean(selected()),
        click: async () => {
          const button = selected()
          if (!button || button.disabled) throw new Error('button not actionable')
          button.click()
        },
      }
      return locator
    },
    waitForTimeout: async ms => vi.setSystemTime(Date.now() + ms),
    evaluate: async fn => window.eval(`(${fn.toString()})()`),
  }
  return { page, clicks, log: vi.fn(), shot: vi.fn() }
}

test.each([true, false])('reveals the actual update button without applying (details=%s)', async details => {
  const f = fixture({ details })
  const update = await updateUi.waitForUpdate(f.page, f)
  expect(await update.isVisible()).toBe(true)
  expect(f.clicks).toEqual(details ? ['Check now', "See what's new"] : ['Check now'])
  expect(f.shot).toHaveBeenCalledWith(f.page, '04-update-available')
})

test('checks the live Desktop bridge against the staged target before opening About', async () => {
  const sha = 'a'.repeat(40)
  const f = fixture({ statusOverride: { supported: true, branch: 'main', currentSha: 'b'.repeat(40), targetSha: sha, behind: 1, dirty: false } })
  await updateUi.assertStagedBranch(f.page, sha, f.log)
  expect(f.log).toHaveBeenCalledWith(expect.stringContaining('"targetSha"'))
  // The current checker's hosted shape: it cannot count staged commits via GitHub compare.
  const current = fixture({ statusOverride: { supported: true, branch: 'main', currentSha: 'b'.repeat(40), targetSha: sha, behind: null, commits: [], updateAvailable: true } })
  await updateUi.assertStagedBranch(current.page, sha, current.log)
  const racing = fixture()
  let checks = 0
  racing.page.evaluate = fn => ++checks === 1
    ? { supported: true, error: 'fetch-failed', message: "error: cannot lock ref 'refs/remotes/origin/main'" }
    : current.page.evaluate(fn)
  await updateUi.assertStagedBranch(racing.page, sha, racing.log)
  expect(checks).toBe(2)
  for (const statusOverride of [
    { supported: true, error: 'release-unavailable', branch: 'main' },
    { supported: true, branch: 'main', targetSha: 'b'.repeat(40), behind: 1, updateAvailable: true },
    { supported: true, branch: 'main', targetSha: sha, behind: 1, dirty: true },
    { supported: true, branch: 'main', targetSha: sha, behind: 0 },
    { supported: true, branch: 'main', currentSha: sha, targetSha: sha, behind: 1 },
    { supported: true, branch: 'main', targetSha: sha, behind: 1, updateAvailable: false },
    { supported: true, branch: 'main', currentSha: 'b'.repeat(40), targetSha: sha, behind: null },
  ]) {
    const refused = fixture({ statusOverride })
    await expect(updateUi.assertStagedBranch(refused.page, sha, refused.log)).rejects.toThrow(/refusing to click/)
  }
})

test('test-only source probe pins staged main without changing other Python invocations', () => {
  const root = '/fixture/install'
  const probe = ['-c', 'from pathlib import Path; import runpy; p = Path("hermes_cli/source_check.py"); entry = runpy.run_path(str(p)).get("main") if p.is_file() else None; entry()', '--install-root', root, '--home', '/fixture/home', '--git', '/shim/git', '--force']
  const expected = [...probe]
  expected[expected.indexOf('--git') + 1] = '/real/git'
  expected.push('--branch', 'main')
  expect(sourceBranchProbe.branchProbeArgs(probe, root, '/real/git')).toEqual(expected)
  const managed = ['--run-module', 'hermes_cli.source_check', ...probe.slice(2)]
  expect(sourceBranchProbe.branchProbeArgs(managed, root, '/real/git')).toEqual([
    '--run-module', 'hermes_cli.source_check', ...expected.slice(2),
  ])
  expect(sourceBranchProbe.branchProbeArgs(probe, '/other/install', '/real/git')).toBe(probe)
  expect(sourceBranchProbe.branchProbeArgs(managed, '/other/install', '/real/git')).toBe(managed)
  expect(sourceBranchProbe.branchProbeArgs(['-c', 'print("hermes_cli/source_check.py")'], root, '/real/git')).toEqual(['-c', 'print("hermes_cli/source_check.py")'])
  expect(sourceBranchProbe.branchProbeArgs(['--run-module', 'hermes_cli.config', ...managed.slice(2)], root, '/real/git')).toEqual(['--run-module', 'hermes_cli.config', ...managed.slice(2)])
  expect(sourceBranchProbe.branchProbeArgs([...probe, '--branch', 'topic'], root, '/real/git')).toEqual([...probe, '--branch', 'topic'])
  expect(sourceBranchProbe.branchProbeArgs(probe, root, '')).toBe(probe)
  const cmd = ['/d', '/v:off', '/s', '/c', `""C:\\fixture\\hermes.cmd" "--run-module" "hermes_cli.source_check" "--install-root" "${root}" "--home" "/profile with spaces" "--git" "/shim/git" "--force""`]
  const rewritten = sourceBranchProbe.branchProbeArgs(cmd, root, '/real/git')
  expect(rewritten.slice(0, 4)).toEqual(cmd.slice(0, 4))
  expect(rewritten[4]).toContain('"--git" "/real/git" "--force" "--branch" "main"')
  expect(sourceBranchProbe.branchProbeArgs(cmd, '/other/install', '/real/git')).toBe(cmd)
  expect(sourceBranchProbe.branchProbeArgs(cmd, root, '/real&git')).toBe(cmd)
})

test.skipIf(process.platform === 'win32')('probe Git reaches the staged main even with global Git config isolated', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-staged-git-'))
  const git = process.env.HERMES_E2E_REAL_GIT || process.env.PATH.split(path.delimiter)
    .map(dir => path.join(dir, process.platform === 'win32' ? 'git.exe' : 'git')).find(file => fs.existsSync(file))
  try {
    const checkout = path.join(root, 'checkout')
    const bare = path.join(root, 'serve.git')
    fs.mkdirSync(checkout)
    const run = (args, cwd = checkout, env = process.env) => execFileSync(git, args, { cwd, env, encoding: 'utf8' }).trim()
    run(['init', '-b', 'main'])
    run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'base'])
    const base = run(['rev-parse', 'HEAD'])
    run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'staged'])
    const sha = run(['rev-parse', 'HEAD'])
    run(['clone', '--bare', checkout, bare], root)
    run(['reset', '--hard', base])
    run(['remote', 'add', 'origin', 'https://github.com/NousResearch/hermes-agent.git'])
    const cfg = path.join(root, 'gitconfig')
    run(['config', '--file', cfg, '--add', `url.file://${bare}.insteadOf`, 'https://github.com/NousResearch/hermes-agent.git'])
    const python = process.env.HERMES_PYTHON || 'python3'
    const source = fileURLToPath(new URL('../', import.meta.url))
    const launcher = path.join(checkout, '.hermes', 'bin', 'hermes')
    fs.mkdirSync(path.dirname(launcher), { recursive: true })
    const quote = value => `'${value.replace(/'/g, "'\\''")}'`
    fs.writeFileSync(launcher, `#!/bin/sh\nif [ "$1" = '--run-module' ]; then shift 2; exec ${quote(python)} -m hermes_cli.source_check "$@"; fi\nprintf '%s\\n' "$@"\n`, { mode: 0o700 })
    const capturedEnv = { ...process.env, GIT_CONFIG_GLOBAL: cfg }
    const launchEnv = { HERMES_DESKTOP_USER_DATA_DIR: root }
    sourceBranchProbe.prepareSourceBranchEnvironment(checkout, sha, git, capturedEnv, launchEnv)
    const env = { ...process.env, GIT_CONFIG_GLOBAL: process.platform === 'win32' ? 'NUL' : '/dev/null',
      PYTHONPATH: source, GIT_ALLOW_PROTOCOL: 'file' }
    const home = path.join(root, 'profile')
    fs.mkdirSync(home)
    const status = JSON.parse(execFileSync(launcher, ['--run-module', 'hermes_cli.source_check',
      '--install-root', checkout, '--home', home, '--git', git, '--force'],
    { cwd: checkout, encoding: 'utf8', env }))
    expect(status).toMatchObject({ supported: true, currentSha: base, branch: 'main', targetSha: sha, updateAvailable: true })
    expect(execFileSync(launcher, ['--version'], { cwd: checkout, env, encoding: 'utf8' }).trim()).toBe('--version')
    const other = path.join(root, 'other-checkout')
    fs.mkdirSync(other)
    const foreign = JSON.parse(execFileSync(launcher, ['--run-module', 'hermes_cli.source_check',
      '--install-root', other, '--home', home, '--git', git, '--force'],
    { cwd: checkout, encoding: 'utf8', env }))
    expect(foreign).toMatchObject({ supported: false, reason: 'not-a-git-checkout' })
    expect(() => sourceBranchProbe.prepareSourceBranchEnvironment(checkout, '0'.repeat(40), git, capturedEnv, launchEnv)).toThrow(/does not match expected/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('historical venv install without a PM launcher still checks staged Git main', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-legacy-branch-'))
  const git = execFileSync('which', ['git'], { encoding: 'utf8' }).trim()
  try {
    const checkout = path.join(root, 'checkout')
    const bare = path.join(root, 'serve.git')
    fs.mkdirSync(checkout)
    const run = (args, cwd = checkout) => execFileSync(git, args, { cwd, encoding: 'utf8' }).trim()
    run(['init', '-b', 'main'])
    run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'old'])
    const old = run(['rev-parse', 'HEAD'])
    run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'target'])
    const sha = run(['rev-parse', 'HEAD'])
    run(['clone', '--bare', checkout, bare], root)
    run(['reset', '--hard', old])
    run(['remote', 'add', 'origin', 'https://github.com/NousResearch/hermes-agent.git'])
    const cfg = path.join(root, 'gitconfig')
    run(['config', '--file', cfg, '--add', `url.file://${bare}.insteadOf`, 'https://github.com/NousResearch/hermes-agent.git'])
    const legacy = path.join(checkout, 'venv', 'bin', 'hermes')
    fs.mkdirSync(path.dirname(legacy), { recursive: true })
    fs.writeFileSync(legacy, '#!/bin/sh\nexit 0\n', { mode: 0o700 })
    const launchEnv = { HERMES_DESKTOP_USER_DATA_DIR: root }
    const capturedEnv = { ...process.env, GIT_CONFIG_GLOBAL: cfg }
    sourceBranchProbe.prepareSourceBranchEnvironment(checkout, sha, git, capturedEnv, launchEnv)
    expect(fs.existsSync(path.join(checkout, '.hermes', 'bin', 'hermes'))).toBe(false)
    expect(launchEnv.HERMES_E2E_SOURCE_ROOT).toBe(checkout)
    expect(launchEnv.HERMES_E2E_SOURCE_URL).toBe(`file://${bare}`)
    expect(launchEnv.NODE_OPTIONS).toContain('source-branch-probe.cjs')
    const launcher = path.join(checkout, '.hermes', 'bin', 'hermes')
    fs.mkdirSync(path.dirname(launcher), { recursive: true })
    fs.symlinkSync(path.join(root, 'missing'), launcher)
    expect(() => sourceBranchProbe.prepareSourceBranchEnvironment(checkout, sha, git, capturedEnv, launchEnv)).toThrow(/launcher/)
    fs.unlinkSync(launcher)
    // A PM tree with no published launcher is unfinished, not legacy.
    fs.mkdirSync(path.join(checkout, 'pm'))
    fs.writeFileSync(path.join(checkout, 'pm', 'lock.json'), '{}')
    expect(() => sourceBranchProbe.prepareSourceBranchEnvironment(checkout, sha, git, { ...process.env, GIT_CONFIG_GLOBAL: cfg }, launchEnv)).toThrow(/launcher/)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('preloaded historical Desktop Git check reads staged origin rather than the public API', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-historical-probe-'))
  const git = execFileSync('which', ['git'], { encoding: 'utf8' }).trim()
  try {
    const checkout = path.join(root, 'checkout')
    const bare = path.join(root, 'serve.git')
    fs.mkdirSync(checkout)
    const run = (args, cwd = checkout) => execFileSync(git, args, { cwd, encoding: 'utf8' }).trim()
    run(['init', '-b', 'main'])
    run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'base'])
    run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '--allow-empty', '-m', 'staged'])
    const sha = run(['rev-parse', 'HEAD'])
    run(['clone', '--bare', checkout, bare], root)
    run(['remote', 'add', 'origin', 'https://github.com/NousResearch/hermes-agent.git'])
    const cfg = path.join(root, 'gitconfig')
    run(['config', '--file', cfg, '--add', `url.file://${bare}.insteadOf`, 'https://github.com/NousResearch/hermes-agent.git'])
    const shim = path.join(root, 'git')
    fs.writeFileSync(shim, `#!/bin/sh\nif [ "$1 $2 $3" = "remote get-url origin" ]; then printf '%s\\n' 'https://github.com/NousResearch/hermes-agent.git'; else exec '${git}' "$@"; fi\n`, { mode: 0o700 })
    const launcher = path.join(checkout, '.hermes', 'bin', 'hermes')
    fs.mkdirSync(path.dirname(launcher), { recursive: true })
    fs.writeFileSync(launcher, '#!/bin/sh\nexit 0\n', { mode: 0o700 })
    const launchEnv = { HERMES_DESKTOP_USER_DATA_DIR: root }
    sourceBranchProbe.prepareSourceBranchEnvironment(checkout, sha, git, { ...process.env, GIT_CONFIG_GLOBAL: cfg }, launchEnv)
    const script = 'const {spawn} = require("node:child_process"); const child = spawn(process.argv[1], process.argv.slice(3), {cwd:process.argv[2], env:{...process.env, GIT_CONFIG_GLOBAL:"/dev/null"}}); child.stdout.pipe(process.stdout); child.stderr.pipe(process.stderr); child.on("close", code => process.exitCode=code)'
    for (const [args, expected] of [
      [['remote', 'get-url', 'origin'], `file://${bare}`],
      [['ls-remote', 'origin', 'refs/heads/main'], `${sha}\trefs/heads/main`],
    ]) {
      const result = spawnSync(process.execPath, ['-e', script, shim, checkout, ...args], {
        encoding: 'utf8', env: { ...process.env, ...launchEnv, GIT_CONFIG_GLOBAL: '/dev/null' },
      })
      expect(result.status, result.stderr).toBe(0)
      expect(result.stdout.trim()).toBe(expected)
    }
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('preloaded Electron-style execFile transports explicit branch into the checker', () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-branch-probe-'))
  try {
    fs.mkdirSync(path.join(root, 'hermes_cli'))
    fs.writeFileSync(path.join(root, 'hermes_cli', 'source_check.py'), 'import json, sys\ndef main(): print(json.dumps(sys.argv[1:]))\n')
    const probe = 'from pathlib import Path; import runpy; p = Path("hermes_cli/source_check.py"); entry = runpy.run_path(str(p)).get("main") if p.is_file() else None; entry() if callable(entry) else print("null")'
    const python = process.env.HERMES_PYTHON || 'python3'
    const script = 'const {promisify} = require("node:util"); const {execFile} = require("node:child_process"); promisify(execFile)(process.argv[1], ["-c", process.argv[2], "--install-root", process.argv[3], "--home", process.argv[3], "--git", "fixture-shim", "--force"], {cwd:process.argv[3]}).then(r=>console.log(r.stdout), e=>{console.error(e);process.exitCode=1})'
    const result = spawnSync(process.execPath, ['-e', script, python, probe, root], {
      encoding: 'utf8', env: { ...process.env, HERMES_E2E_SOURCE_ROOT: root, HERMES_E2E_SOURCE_GIT: '/real/git',
        NODE_OPTIONS: `--require=${JSON.stringify(fileURLToPath(new URL('../tests/install/e2e-assets/source-branch-probe.cjs', import.meta.url)))}` },
    })
    expect(result.status, result.stderr).toBe(0)
    expect(JSON.parse(result.stdout)).toEqual(['--install-root', root, '--home', root, '--git', '/real/git', '--force', '--branch', 'main'])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('does not turn a completed check without an update into success', async () => {
  const f = fixture({ available: false })
  await expect(updateUi.waitForUpdate(f.page, f)).rejects.toThrow(/"Update now" never appeared/)
  expect(f.clicks).toEqual(['Check now'])
  expect(f.log).toHaveBeenCalledWith('[update-status] {"supported":true,"behind":0}')
  expect(f.shot).toHaveBeenCalledWith(f.page, 'ERROR-no-update-now')
})

function initCaseCollisionRepo(root, entries, { ignoreCase = true } = {}) {
  const git = process.env.HERMES_E2E_REAL_GIT || 'git'
  const run = (args, opts = {}) => execFileSync(git, ['-C', root, ...args], {
    encoding: 'utf8',
    ...opts,
  }).trim()
  fs.mkdirSync(root, { recursive: true })
  run(['init', '-b', 'main'])
  run(['config', 'core.autocrlf', 'false'])
  run(['config', 'core.ignorecase', ignoreCase ? 'true' : 'false'])
  for (const { mode = '100644', content, path: relPath } of entries) {
    const oid = execFileSync(git, ['-C', root, 'hash-object', '-w', '--stdin'], {
      input: Buffer.isBuffer(content) ? content : Buffer.from(content),
      encoding: 'utf8',
    }).trim()
    run(['update-index', '--add', '--cacheinfo', `${mode},${oid},${relPath}`])
  }
  run(['-c', 'user.name=Fixture', '-c', 'user.email=e2e@example.invalid', '-c', 'commit.gpgsign=false', 'commit', '-m', 'base'])
  run(['checkout', '--force', 'HEAD'])
  return { git, run }
}

test('assertStagedBranch tolerates dirty=true only when repoDir has verified HEAD case collisions', async () => {
  const sha = 'a'.repeat(40)
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-case-collision-'))
  try {
    const upper = 'contributors/emails/agent@Agents-Mac-mini.local'
    const lower = 'contributors/emails/agent@agents-Mac-mini.local'
    const { git, run } = initCaseCollisionRepo(dir, [
      { path: 'tracked.txt', content: 'v1\n' },
      { path: upper, content: 'skip-agent\n' },
      { path: lower, content: 'momomojo\n' },
    ])
    fs.writeFileSync(path.join(dir, upper), 'momomojo\n')

    const dirtyStatus = {
      supported: true, branch: 'main', currentSha: 'b'.repeat(40),
      targetSha: sha, behind: 1, dirty: true,
    }
    const ok = fixture({ statusOverride: dirtyStatus })
    await updateUi.assertStagedBranch(ok.page, sha, ok.log, { repoDir: dir, gitExe: git })
    expect(ok.log).toHaveBeenCalledWith(expect.stringContaining('tolerating dirty=true'))

    fs.writeFileSync(path.join(dir, upper), 'momomojo\r\n')
    const crlf = fixture({ statusOverride: dirtyStatus })
    await updateUi.assertStagedBranch(crlf.page, sha, crlf.log, { repoDir: dir, gitExe: git })

    const wrongTarget = fixture({ statusOverride: { ...dirtyStatus, targetSha: 'c'.repeat(40) } })
    await expect(updateUi.assertStagedBranch(wrongTarget.page, sha, wrongTarget.log, { repoDir: dir, gitExe: git }))
      .rejects.toThrow(/refusing to click/)

    fs.writeFileSync(path.join(dir, upper), 'tampered\n')
    const tampered = fixture({ statusOverride: dirtyStatus })
    await expect(updateUi.assertStagedBranch(tampered.page, sha, tampered.log, { repoDir: dir, gitExe: git }))
      .rejects.toThrow(/refusing to click/)

    fs.writeFileSync(path.join(dir, upper), 'momomojo\n')
    fs.writeFileSync(path.join(dir, 'tracked.txt'), 'edited\n')
    const extraDirty = fixture({ statusOverride: dirtyStatus })
    await expect(updateUi.assertStagedBranch(extraDirty.page, sha, extraDirty.log, { repoDir: dir, gitExe: git }))
      .rejects.toThrow(/refusing to click/)
    fs.writeFileSync(path.join(dir, 'tracked.txt'), 'v1\n')

    fs.writeFileSync(path.join(dir, 'stray.txt'), 'untracked\n')
    const untracked = fixture({ statusOverride: dirtyStatus })
    await expect(updateUi.assertStagedBranch(untracked.page, sha, untracked.log, { repoDir: dir, gitExe: git }))
      .rejects.toThrow(/refusing to click/)
    fs.unlinkSync(path.join(dir, 'stray.txt'))

    run(['-c', 'core.ignorecase=false', 'add', upper])
    const staged = fixture({ statusOverride: dirtyStatus })
    await expect(updateUi.assertStagedBranch(staged.page, sha, staged.log, { repoDir: dir, gitExe: git }))
      .rejects.toThrow(/refusing to click/)
    run(['reset', '-q', 'HEAD'])

    run(['config', 'core.ignorecase', 'false'])
    const caseSensitive = fixture({ statusOverride: dirtyStatus })
    await expect(updateUi.assertStagedBranch(caseSensitive.page, sha, caseSensitive.log, { repoDir: dir, gitExe: git }))
      .rejects.toThrow(/refusing to click/)

    const cleanDir = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-clean-'))
    try {
      const clean = initCaseCollisionRepo(cleanDir, [{ path: 'a.txt', content: 'ok\n' }])
      const mismatch = fixture({ statusOverride: dirtyStatus })
      await expect(updateUi.assertStagedBranch(mismatch.page, sha, mismatch.log, { repoDir: cleanDir, gitExe: clean.git }))
        .rejects.toThrow(/refusing to click/)
    } finally {
      fs.rmSync(cleanDir, { recursive: true, force: true })
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('case-collision helper handles spaces, renames, binary blobs, and CLI modes', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'case-collision-edge-'))
  try {
    const cli = fileURLToPath(new URL('../tests/install/e2e-assets/case-collision.cjs', import.meta.url))
    const upperSpace = 'dir with spaces/File Name.txt'
    const lowerSpace = 'dir with spaces/file name.txt'
    const upperBin = 'bin/Data.bin'
    const lowerBin = 'bin/data.bin'
    const { git, run } = initCaseCollisionRepo(dir, [
      { path: upperSpace, content: 'alpha\n' },
      { path: lowerSpace, content: 'beta\n' },
      { path: upperBin, content: Buffer.from([0x00, 0x01, 0x0a]) },
      { path: lowerBin, content: Buffer.from([0x00, 0x02, 0x0a]) },
      { path: 'rename-src.txt', content: 'moved\n' },
    ])

    fs.writeFileSync(path.join(dir, upperSpace), 'beta\n')
    fs.writeFileSync(path.join(dir, upperBin), Buffer.from([0x00, 0x02, 0x0a]))
    expect(caseCollision.inspectCaseCollisionDirty(dir, git)).toEqual({
      ignoreCase: true,
      collisions: expect.arrayContaining([upperSpace, upperBin]),
      other: [],
    })
    const onlyOk = spawnSync(process.execPath, [cli, '--only-collisions', dir, '--git', git], { encoding: 'utf8' })
    expect(onlyOk.status, onlyOk.stderr).toBe(0)
    const listed = spawnSync(process.execPath, [cli, '--list', dir, '--git', git], { encoding: 'utf8' })
    expect(listed.status, listed.stderr).toBe(0)
    expect(JSON.parse(listed.stdout).other).toEqual([])

    fs.writeFileSync(path.join(dir, upperBin), Buffer.from([0x00, 0x02, 0x0d, 0x0a]))
    const binBad = caseCollision.inspectCaseCollisionDirty(dir, git)
    expect(binBad.other.length).toBe(1)
    expect(spawnSync(process.execPath, [cli, '--only-collisions', dir, '--git', git]).status).toBe(1)
    fs.writeFileSync(path.join(dir, upperBin), Buffer.from([0x00, 0x02, 0x0a]))

    run(['mv', 'rename-src.txt', 'rename-dst.txt'])
    const renamed = caseCollision.inspectCaseCollisionDirty(dir, git)
    expect(renamed.other.length).toBe(1)
    expect(renamed.other[0]).toContain('rename-dst.txt')
    expect(renamed.other[0]).toContain('rename-src.txt')
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test.skipIf(process.platform === 'win32')('case-collision helper never follows a symlink for a regular-file sibling', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'case-collision-symlink-'))
  try {
    const upper = 'docs/Note.txt'
    const lower = 'docs/note.txt'
    const { git } = initCaseCollisionRepo(dir, [
      { path: upper, content: 'one\n' },
      { path: lower, content: 'two\n' },
    ])
    const external = path.join(dir, '..', `external-${path.basename(dir)}.txt`)
    fs.writeFileSync(external, 'two\n')
    try {
      fs.unlinkSync(path.join(dir, upper))
      fs.symlinkSync(external, path.join(dir, upper))
      const res = caseCollision.inspectCaseCollisionDirty(dir, git)
      expect(res.collisions).toEqual([])
      expect(res.other.length).toBe(1)
    } finally {
      fs.rmSync(external, { force: true })
    }
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})