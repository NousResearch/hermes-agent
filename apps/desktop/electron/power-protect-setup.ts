import { execFile } from 'node:child_process'
import { promises as fs, mkdtempSync, writeFileSync } from 'node:fs'
import os from 'node:os'
import path from 'node:path'

export interface PowerProtectSetupResult {
  error?: string
  ok: boolean
}

const TARGET = '/etc/sudoers.d/hermes-power-protect'
const RULE = `Cmnd_Alias PMSET_HERMES_POWER = /usr/bin/pmset -a disablesleep 1, /usr/bin/pmset -a disablesleep 0\n%admin ALL=(ALL) NOPASSWD: PMSET_HERMES_POWER\n`

type ExecFile = typeof execFile

function appleLiteral(value: string): string {
  return JSON.stringify(value)
}

function run(
  exec: ExecFile,
  command: string,
  args: string[]
): Promise<{ code: number; stderr: string; stdout: string }> {
  return new Promise(resolve => {
    exec(command, args, { encoding: 'utf8' }, (error, stdout, stderr) => {
      resolve({
        code: error ? (typeof error.code === 'number' ? error.code : 1) : 0,
        stderr: String(stderr || ''),
        stdout: String(stdout || '')
      })
    })
  })
}

function hasExactRule(output: string): boolean {
  return (
    output.includes('NOPASSWD') &&
    output.includes('/usr/bin/pmset -a disablesleep 1') &&
    output.includes('/usr/bin/pmset -a disablesleep 0')
  )
}

async function isInstalled(exec: ExecFile): Promise<boolean> {
  try {
    const stat = await fs.lstat(TARGET)

    if (!stat.isFile() || stat.uid !== 0 || (stat.mode & 0o777) !== 0o440) {
      return false
    }
  } catch {
    return false
  }

  const verified = await run(exec, '/usr/bin/sudo', ['-n', '-l'])

  return verified.code === 0 && hasExactRule(`${verified.stdout}\n${verified.stderr}`)
}

/** Install the fixed Hermes sudoers rule through macOS's native admin prompt. */
export async function installPowerProtect(
  exec: ExecFile = execFile,
  platformName: NodeJS.Platform = process.platform
): Promise<PowerProtectSetupResult> {
  if (platformName !== 'darwin') {
    return { ok: false, error: 'Power Protect setup is only available on macOS.' }
  }

  if (await isInstalled(exec)) {
    return { ok: true }
  }

  const tempDir = mkdtempSync(path.join(os.tmpdir(), 'hermes-power-protect-'))
  const source = path.join(tempDir, 'hermes-power-protect')
  writeFileSync(source, RULE, { encoding: 'utf8', mode: 0o600 })

  try {
    const sourceLiteral = appleLiteral(source.toString())
    const targetLiteral = appleLiteral(TARGET)

    const script = [
      `set p to quoted form of ${sourceLiteral}`,
      `set target to quoted form of ${targetLiteral}`,
      'do shell script "/usr/sbin/visudo -c -f " & p & " && /usr/bin/install -o root -g wheel -m 0440 " & p & " " & target with administrator privileges'
    ].join('\n')

    const auth = await run(exec, '/usr/bin/osascript', ['-e', script])

    if (auth.code !== 0) {
      return { ok: false, error: auth.stderr.trim() || auth.stdout.trim() || 'Administrator setup was cancelled.' }
    }

    const verified = await run(exec, '/usr/bin/sudo', ['-n', '-l'])
    const output = `${verified.stdout}\n${verified.stderr}`

    if (verified.code !== 0 || !hasExactRule(output)) {
      return { ok: false, error: 'The Hermes Power Protect sudoers rule could not be verified.' }
    }

    return { ok: true }
  } finally {
    await fs.rm(tempDir, { recursive: true, force: true }).catch(() => undefined)
  }
}
