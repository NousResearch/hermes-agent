import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { buildSync } from 'esbuild'
import { expect, test, vi } from 'vitest'

import { createAppRelauncher, relaunchApp } from './app-relaunch'
import { LAUNCHER_READY_FD_ENV } from './linux-launcher-ready'

test('a self-relaunch never hands the launcher ready-pipe fd number to the new process', () => {
  const env: NodeJS.ProcessEnv = { [LAUNCHER_READY_FD_ENV]: '4', KEEP: '1' }
  const seen: Array<{ args?: string[]; carried: string | undefined }> = []

  relaunchApp(
    { relaunch: options => seen.push({ args: options.args, carried: env[LAUNCHER_READY_FD_ENV] }) },
    { args: ['--disable-gpu'] },
    env
  )

  expect(seen).toEqual([{ args: ['--disable-gpu'], carried: undefined }])
  expect(env.KEEP).toBe('1')
})

function recordingApp(events: string[]) {
  return {
    commandLine: { hasSwitch: () => false },
    relaunch: (options: { args?: string[] }) => events.push(`relaunch ${options.args?.join(' ')}`),
    releaseSingleInstanceLock: () => events.push('release'),
    exit: (code: number) => events.push(`exit ${code}`)
  }
}

function noNewPrivs(statusText: string): string | undefined {
  return /^NoNewPrivs:\s*(\d)/m.exec(statusText)?.[1]
}

// What main.ts does around a relaunch, in its own process: request it, then exit after the
// backend wait or die during it. Chromium's fatal "GPU process isn't usable" abort can land in
// that wait, and app.relaunch() comes back from it because its helper starts at once.
const OLD_PROCESS = `
import { createAppRelauncher, relaunchApp } from './app-relaunch'

const [exe, death] = process.argv.slice(2)
const events: string[] = []
const relauncher = createAppRelauncher(
  {
    commandLine: { hasSwitch: () => false },
    relaunch: () => events.push('electron relaunch'),
    releaseSingleInstanceLock: () => events.push('release'),
    exit: code => {
      process.stdout.write([...events, 'exit ' + code].join(','))
      process.exit(code)
    }
  },
  'linux',
  exe,
  process.env
)

relaunchApp(relauncher, { args: ['--disable-gpu'] })
if (death === 'exit') relauncher.exit(0)
else process.kill(process.pid, 'SIGKILL')
`

// Electron's Linux relauncher sets PR_SET_NO_NEW_PRIVS on the new process; the backend and
// the agent's terminal inherit it and sudo refuses. The new process must be an ordinary child.
test.skipIf(process.platform !== 'linux')(
  'a Linux relaunch comes back after an exit or a hard death, unlocked, in its own session, without no_new_privs',
  async () => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-relaunch-'))
    const exe = path.join(dir, 'hermes')
    const oldProcess = path.join(dir, 'old-process.cjs')

    fs.writeFileSync(
      exe,
      `#!/bin/sh\nprintf '%s\\n' "$*" "$(cut -d' ' -f6 /proc/$$/stat) $$" "$(grep NoNewPrivs /proc/$$/status)" > "$REPORT.part"\nmv "$REPORT.part" "$REPORT"\n`,
      { mode: 0o755 }
    )
    buildSync({
      stdin: { contents: OLD_PROCESS, resolveDir: import.meta.dirname, loader: 'ts' },
      outfile: oldProcess,
      bundle: true,
      platform: 'node',
      format: 'cjs',
      external: ['electron'],
      logLevel: 'silent'
    })

    try {
      for (const [death, events] of [
        ['exit', 'release,exit 0'],
        ['kill', '']
      ]) {
        const report = path.join(dir, `report-${death}`)

        const run = spawnSync(process.execPath, [oldProcess, exe, death], {
          encoding: 'utf8',
          env: { PATH: process.env.PATH, REPORT: report, [LAUNCHER_READY_FD_ENV]: '4' }
        })

        expect(run.stdout).toBe(events)
        await vi.waitFor(() => expect(fs.existsSync(report)).toBe(true), { timeout: 5000 })
        const [args, session, privs] = fs.readFileSync(report, 'utf8').split('\n')
        const [sid, pid] = session.split(' ')

        expect(args).toBe('--disable-gpu')
        expect(sid).toBe(pid)
        expect(noNewPrivs(privs)).toBe(noNewPrivs(fs.readFileSync('/proc/self/status', 'utf8')))
      }
    } finally {
      fs.rmSync(dir, { recursive: true, force: true })
    }
  }
)

// Node's teardown closes the helper's stdin before Chromium has shut down (storage flush,
// DevTools, a core dump): a target started then binds next to a live old process and loses
// whatever that still holds, such as the DevTools port.
const LINGERING_PROCESS = `
import { spawnRelaunch } from './relaunch-target'

process.env.OLD_PID = String(process.pid)
spawnRelaunch(process.argv[2], [], process.env).stdin!.end()
setTimeout(() => process.exit(0), 1000)
`

test.skipIf(process.platform !== 'linux')('a Linux relaunch starts only once the old process is dead', async () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-relaunch-linger-'))
  const exe = path.join(dir, 'hermes')
  const oldProcess = path.join(dir, 'old-process.cjs')
  const report = path.join(dir, 'report')

  fs.writeFileSync(
    exe,
    `#!/bin/sh\ns=$(cut -d' ' -f3 /proc/$OLD_PID/stat 2>/dev/null)\necho "\${s:-gone}" > "$REPORT.part"\nmv "$REPORT.part" "$REPORT"\n`,
    { mode: 0o755 }
  )
  buildSync({
    stdin: { contents: LINGERING_PROCESS, resolveDir: import.meta.dirname, loader: 'ts' },
    outfile: oldProcess,
    bundle: true,
    platform: 'node',
    format: 'cjs',
    external: ['electron'],
    logLevel: 'silent'
  })

  try {
    spawnSync(process.execPath, [oldProcess, exe], { env: { PATH: process.env.PATH, REPORT: report } })
    await vi.waitFor(() => expect(fs.existsSync(report)).toBe(true), { timeout: 5000 })
    // Dead: reaped already, or a zombie its parent has not collected yet.
    expect(['gone', 'Z']).toContain(fs.readFileSync(report, 'utf8').trim())
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

test('macOS and Windows keep Electron app.relaunch', () => {
  for (const platform of ['darwin', 'win32']) {
    const events: string[] = []
    const relauncher = createAppRelauncher(recordingApp(events), platform, '/nonexistent/hermes', {})

    relaunchApp(relauncher, { args: ['--no-sandbox'] }, {})
    relauncher.exit(0)

    expect(events).toEqual(['relaunch --no-sandbox', 'exit 0'])
  }
})

// The exit path runs after the backend is down: a target that cannot start falls back to
// Electron's relauncher. JS keeps running briefly after app.exit(), and a relaunch requested
// then (a second Restart) must not start another process.
test.skipIf(process.platform !== 'linux')('a Linux relaunch whose target cannot start uses Electron, once', () => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-relaunch-noexec-'))
  const exe = path.join(dir, 'hermes')
  const events: string[] = []
  const logged = vi.spyOn(console, 'error').mockImplementation(() => {})

  fs.writeFileSync(exe, '', { mode: 0o644 })

  try {
    const relauncher = createAppRelauncher(recordingApp(events), 'linux', exe, { PATH: process.env.PATH })

    relaunchApp(relauncher, { args: ['--disable-gpu'] }, {})
    relauncher.exit(0)
    relaunchApp(relauncher, { args: ['--no-sandbox'] }, {})
    relauncher.exit(0)

    expect(events).toEqual(['relaunch --disable-gpu', 'exit 0', 'exit 0'])
    expect(logged).toHaveBeenCalledTimes(1)
  } finally {
    logged.mockRestore()
    fs.rmSync(dir, { recursive: true, force: true })
  }
})
