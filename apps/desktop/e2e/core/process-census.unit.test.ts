import { spawn, spawnSync } from 'node:child_process'
import * as fs from 'node:fs'
import * as os from 'node:os'
import * as path from 'node:path'

import { afterEach, expect, test, vi } from 'vitest'

afterEach(() => {
  vi.unstubAllEnvs()
  vi.resetModules()
})

// The terminal tool runs commands through a POSIX shell (bash; Git Bash on Windows).
test.skipIf(process.platform === 'win32')('shellQuote keeps a path with spaces and quotes one argument', async () => {
  const { shellQuote } = await import('./process-census')
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "core census o'k "))
  const victim = path.join(dir, 'victim run')
  fs.mkdirSync(victim)
  const bystander = path.join(dir, 'victim')
  fs.mkdirSync(bystander)

  try {
    expect(spawnSync('sh', ['-c', `rm -rf ${shellQuote(victim)}`]).status).toBe(0)
    expect(fs.existsSync(victim)).toBe(false)
    expect(fs.existsSync(bystander), 'an unquoted path split at the space would hit this').toBe(true)
  } finally {
    fs.rmSync(dir, { recursive: true, force: true })
  }
})

// Exercises the non-/proc census (Windows/macOS shape) through `ps`, on the same real processes.
test.skipIf(process.platform === 'win32').each(['procfs', 'ps'])(
  'sandbox census finds the recorded gateway and its children, and drops them once gone (%s)',
  async mode => {
    if (mode === 'procfs' && !fs.existsSync('/proc/self/environ')) {
      return
    }

    vi.stubEnv('HERMES_E2E_PROC_CENSUS', mode === 'ps' ? 'ps' : '')
    const { sandboxProcessesOf, processesTagged } = await import('./process-census')
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'core census home '))
    const tag = `core-census-${process.pid}-${mode}`
    // A stand-in gateway whose child is tagged; the child's environment carries the home too.
    const gateway = spawn('bash', ['-c', '(exec -a "$CORE_TAG" sleep 60) & wait'], {
      env: { ...process.env, HERMES_HOME: home, CORE_TAG: tag },
      stdio: 'ignore'
    })
    fs.writeFileSync(path.join(home, 'gateway.lock'), JSON.stringify({ pid: gateway.pid }))

    try {
      await expect.poll(() => processesTagged(tag).length).toBe(1)
      const census = sandboxProcessesOf(home).map(proc => proc.pid)
      expect(census).toContain(gateway.pid)
      expect(census).toContain(processesTagged(tag)[0].pid)
      expect(sandboxProcessesOf(path.join(home, 'other')).map(proc => proc.pid)).not.toContain(gateway.pid)
    } finally {
      for (const proc of processesTagged(tag)) {
        process.kill(proc.pid, 'SIGKILL')
      }

      gateway.kill('SIGKILL')
      await expect.poll(() => sandboxProcessesOf(home).length).toBe(0)
      fs.rmSync(home, { recursive: true, force: true })
    }
  }
)
