import assert from 'node:assert/strict'
import { execFile as execFileCallback, spawn, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import { REMOTE_MARKER_GATE_PY, REMOTE_MARKER_JUDGE_PY, WINDOWS_MARKER_JUDGE_PS } from './remote-update-marker-programs'
import { parseUpdateMarker } from './update-marker-judge'

const execFile = promisify(execFileCallback)
const corpusPath = path.resolve(__dirname, '../../../tests/fixtures/update_marker_corpus.json')

// Feed every corpus judge case, with its injected process table, through the
// exact Python an SSH remote runs: the remote reader obeys the shared contract.
const DRIVER = String.raw`
import json
corpus=json.load(open(sys.argv[1],encoding='utf-8'))
out={}
for case in corpus['judge']:
    live={int(pid):ct for pid,ct in case['live'].items()}
    own=case.get('our_ct',corpus['our_ct'])
    env={'our_pid':case.get('our_pid',corpus['our_pid']),'our_ct':lambda own=own:own,'alive':lambda pid,live=live:pid in live,'ct':lambda pid,live=live:live.get(pid),'now':corpus['now']}
    verdict,owner=marker_judge(case['text'],env)
    out[case['name']]={'verdict':verdict,'owner':owner}
print(json.dumps(out))
`

test.skipIf(process.platform === 'win32')('the remote marker judge agrees with every corpus judge case', async () => {
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'))
  const { stdout } = await execFile('python3', ['-c', `${REMOTE_MARKER_JUDGE_PY}\n${DRIVER}`, corpusPath])

  const expected = Object.fromEntries(
    corpus.judge.map((c: any) => [c.name, { verdict: c.expect.verdict, owner: c.expect.owner }])
  )

  assert.ok(corpus.judge.length >= 40)
  assert.deepEqual(JSON.parse(stdout), expected)
})

// The same replay through the PowerShell judge Windows remotes run, with each
// case's live table and the corpus clock injected over its fact seams.
const PS_DRIVER = String.raw`
function Get-MarkerProcessFacts([int]$ownerId){if($script:live.ContainsKey([int64]$ownerId)){return @{Alive=$true;Ct=$script:live[[int64]$ownerId]}};return @{Alive=$false}}
function Get-MarkerNow{[int64]$script:corpus.now}
$script:corpus=Get-Content -LiteralPath $args[0] -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($case in $script:corpus.judge){
$script:live=@{};foreach($p in $case.live.PSObject.Properties){$script:live[[int64]$p.Name]=if($null -eq $p.Value){$null}else{[double]$p.Value}}
[Console]::Out.WriteLine($case.name+[char]9+(Get-MarkerVerdict $case.text))
}
`

const powershell = ['pwsh', 'powershell'].find(shell => spawnSync(shell, ['-NoProfile', '-Command', 'exit 0']).status === 0)

test.skipIf(!powershell)('the Windows remote marker judge agrees with every corpus judge case', async () => {
  const corpus = JSON.parse(readFileSync(corpusPath, 'utf8'))
  const dir = mkdtempSync(path.join(os.tmpdir(), 'hermes-remote-ps-judge-'))
  const script = path.join(dir, 'replay.ps1')

  try {
    writeFileSync(script, `${WINDOWS_MARKER_JUDGE_PS}\n${PS_DRIVER}`)
    const { stdout } = await execFile(powershell!, ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', script, corpusPath])
    const verdicts = Object.fromEntries(stdout.trim().split(/\r?\n/).map(line => line.split('\t')))

    // Only the host shell is "us" to this judge (never a marker owner), so the
    // corpus cases about the reader's own pid do not apply.
    const cases = corpus.judge.filter((c: any) => {
      const marker = parseUpdateMarker(c.text)

      return ![marker?.pid, marker?.delegate?.pid].includes(c.our_pid ?? corpus.our_pid)
    })

    const verdict = (c: any) =>
      ({ malformed: 'UNCERTAIN', dead: 'CLEAR', live: `LIVE:${c.expect.owner}` })[c.expect.verdict as string]

    assert.ok(cases.length >= 40)
    assert.deepEqual(
      Object.fromEntries(cases.map((c: any) => [c.name, verdicts[c.name]])),
      Object.fromEntries(cases.map((c: any) => [c.name, verdict(c)]))
    )
  } finally {
    rmSync(dir, { recursive: true, force: true })
  }
  // pwsh cold start alone can take seconds on a loaded runner; same budget as the other subprocess tests.
}, 20_000)

// A dead claim whose checkout lock is still flocked (a killed updater's completion
// child) must be kept: the gate answers HELD instead of unlinking it (review G1).
const HOLD_PY = String.raw`
import fcntl,os,sys,time
fd=os.open(sys.argv[1],os.O_RDWR|os.O_CREAT)
fcntl.flock(fd,fcntl.LOCK_EX)
print('held',flush=True)
time.sleep(60)
`

async function withCheckoutLockHeld(lockPath: string, body: () => Promise<void>) {
  const holder = spawn('python3', ['-c', HOLD_PY, lockPath], { stdio: ['ignore', 'pipe', 'inherit'] })

  try {
    await new Promise(resolve => holder.stdout.once('data', resolve))
    await body()
  } finally {
    holder.kill('SIGKILL')
    await new Promise(resolve => holder.once('exit', resolve))
  }
}

function deadMarker(marker: string) {
  const exited = spawnSync('true')
  writeFileSync(marker, `${exited.pid}\n${Math.floor(Date.now() / 1000)}\n`)
}

async function gate(args: string[]) {
  try {
    const { stdout, stderr } = await execFile('python3', ['-c', REMOTE_MARKER_GATE_PY, ...args])

    return { code: 0, stdout: stdout.trim(), stderr: stderr.trim() }
  } catch (error: any) {
    return { code: error.code, stdout: String(error.stdout).trim(), stderr: String(error.stderr).trim() }
  }
}

test.skipIf(process.platform === 'win32')(
  'the remote gate keeps a dead marker while the default checkout lock is held',
  async () => {
    const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-remote-gate-'))
    const marker = path.join(root, '.hermes-update-in-progress')

    try {
      mkdirSync(path.join(root, 'hermes-agent', '.git'), { recursive: true })
      deadMarker(marker)

      await withCheckoutLockHeld(path.join(root, 'hermes-agent', '.git', 'hermes-update.lock'), async () => {
        assert.deepEqual(await gate([marker]), { code: 0, stdout: 'HELD', stderr: '' })
        assert.ok(existsSync(marker), 'a dead marker beside a held checkout lock must be kept')
        const refused = await gate([marker, 'echo PAYLOAD-RAN'])
        assert.deepEqual(refused, { code: 75, stdout: '', stderr: 'HELD' })
      })

      assert.deepEqual(await gate([marker]), { code: 0, stdout: 'CLEAR', stderr: '' })
      assert.ok(!existsSync(marker), 'a dead marker beside a free checkout lock is reclaimed')
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  }
)

test.skipIf(process.platform === 'win32')(
  'the remote gate probes the checkout of the hermes executable it is given (linked worktree)',
  async () => {
    const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-remote-gate-'))
    const home = path.join(root, 'home')
    const marker = path.join(home, '.hermes-update-in-progress')
    const checkout = path.join(root, 'src', 'hermes-agent')
    const common = path.join(root, 'repo.git')

    try {
      mkdirSync(home, { recursive: true })
      mkdirSync(path.join(checkout, 'venv', 'bin'), { recursive: true })
      mkdirSync(path.join(common, 'worktrees', 'co'), { recursive: true })
      writeFileSync(path.join(checkout, '.git'), `gitdir: ${path.join(common, 'worktrees', 'co')}\n`)
      writeFileSync(path.join(common, 'worktrees', 'co', 'commondir'), '../..\n')
      writeFileSync(path.join(checkout, 'venv', 'bin', 'hermes'), '')
      mkdirSync(path.join(root, 'bin'))
      symlinkSync(path.join(checkout, 'venv', 'bin', 'hermes'), path.join(root, 'bin', 'hermes'))
      deadMarker(marker)

      await withCheckoutLockHeld(path.join(common, 'hermes-update.lock'), async () => {
        const hermes = path.join(root, 'bin', 'hermes')
        assert.deepEqual(await gate([marker, '', hermes]), { code: 0, stdout: 'HELD', stderr: '' })
        assert.deepEqual((await gate([marker, 'echo PAYLOAD-RAN', hermes])).code, 75)
        assert.ok(existsSync(marker))
      })
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  }
)

// macOS/BSD remotes read creation time from `ps -o lstart=`. Python's judge asks
// for it in UTC (TZ=UTC0 + calendar.timegm); a local-time parse of the repeated
// DST hour lands an hour off and judges a live owner dead (review G2).
test.skipIf(process.platform === 'win32')('the remote judge reads ps lstart in UTC, not local time', async () => {
  const root = mkdtempSync(path.join(os.tmpdir(), 'hermes-remote-lstart-'))

  try {
    // Started 01:30 EST on 2026-11-01 (06:30 UTC): New York local time repeats 01:30.
    writeFileSync(
      path.join(root, 'ps'),
      '#!/bin/sh\nif [ "$TZ" = UTC0 ]; then echo "Sun Nov  1 06:30:00 2026"; else echo "Sun Nov  1 01:30:00 2026"; fi\n',
      { mode: 0o755 }
    )

    const driver = `${REMOTE_MARKER_JUDGE_PY}\nsys.platform='darwin'\nprint(repr(marker_ct(4242)))`

    const { stdout } = await execFile('python3', ['-c', driver], {
      env: { ...process.env, PATH: `${root}:${process.env.PATH}`, TZ: 'America/New_York' }
    })

    assert.equal(Number(stdout.trim()), Date.UTC(2026, 10, 1, 6, 30) / 1000)
  } finally {
    rmSync(root, { recursive: true, force: true })
  }
})
