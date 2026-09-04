import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { resolveBashExecutable } from './bash-resolver'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const POSIX_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'posix.sh')
const WINDOWS_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'windows.ps1')

function sandbox(tag: string) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-handoff-marker-${tag}-`))
  const installRoot = path.join(home, 'hermes-agent')
  fs.mkdirSync(installRoot)

  return { home, installRoot }
}

function markerStartedAt(home: string): number {
  const [, startedAt] = fs.readFileSync(path.join(home, '.hermes-update-in-progress'), 'utf8').split('\n')

  return Number.parseInt(startedAt, 10)
}

function runPosix(installRoot: string, startedAt?: string) {
  const env = { ...process.env }

  if (startedAt === undefined) {
    delete env.HERMES_UPDATE_STARTED_AT
  } else {
    env.HERMES_UPDATE_STARTED_AT = startedAt
  }

  return spawnSync(
    resolveBashExecutable() ?? 'bash',
    [POSIX_SCRIPT, '--daemonized', '--install-root', installRoot, '--self-test-marker'],
    {
      env,
      encoding: 'utf8'
    }
  )
}

function executableOnPath(name: string): string {
  for (const directory of (process.env.PATH || '').split(path.delimiter)) {
    if (!directory) {
      continue
    }

    const candidate = path.join(directory, name)

    try {
      const stat = fs.statSync(candidate)

      if (stat.isFile() && (stat.mode & 0o111) !== 0) {
        return fs.realpathSync(candidate)
      }
    } catch {
      // Keep looking.
    }
  }

  throw new Error(`could not resolve ${name} from PATH`)
}

function resolveHostPython(): string {
  for (const name of ['python3', 'python']) {
    try {
      return executableOnPath(name)
    } catch {
      // Try the next candidate.
    }
  }

  // Some development environments ship no system python; fall back to a uv-managed one.
  const uvLookup = spawnSync(executableOnPath('uv'), ['python', 'find'], { encoding: 'utf8' })
  const discovered = String(uvLookup.stdout || '').trim()

  assert.ok(
    discovered,
    String(uvLookup.stderr || 'no python3, python, or uv-managed interpreter available on PATH')
  )

  return discovered
}

test.skipIf(process.platform === 'win32')('POSIX hand-off resolves re-exec helpers from PATH', async () => {
  const { home, installRoot } = sandbox('path-helpers')
  const mockBin = path.join(home, 'mock-bin')
  const realBash = executableOnPath('bash')
  const realSh = executableOnPath('sh')
  const realPython = resolveHostPython()
  const helperLog = path.join(home, 'helpers.log')

  fs.mkdirSync(mockBin)

  for (const command of ['cat', 'date', 'dirname', 'mkdir', 'pwd', 'rm', 'tee']) {
    fs.symlinkSync(executableOnPath(command), path.join(mockBin, command))
  }

  fs.writeFileSync(
    path.join(mockBin, 'bash'),
    `#!${realSh}\nprintf 'bash\\n' >> ${JSON.stringify(helperLog)}\nexec ${JSON.stringify(realBash)} "$@"\n`,
    { mode: 0o755 }
  )
  fs.writeFileSync(
    path.join(mockBin, 'nohup'),
    `#!${realSh}\nprintf 'nohup\\n' >> ${JSON.stringify(helperLog)}\nexec "$@"\n`,
    { mode: 0o755 }
  )
  fs.writeFileSync(
    path.join(mockBin, 'python3'),
    `#!${realSh}\nprintf 'python3\\n' >> ${JSON.stringify(helperLog)}\nexec ${JSON.stringify(realPython)} "$@"\n`,
    { mode: 0o755 }
  )

  const result = spawnSync(realBash, [POSIX_SCRIPT, '--install-root', installRoot, '--self-test-marker'], {
    env: { ...process.env, PATH: mockBin },
    encoding: 'utf8'
  })

  assert.equal(result.status, 0, String(result.stderr || result.stdout))

  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      if (fs.readFileSync(helperLog, 'utf8').includes('bash\n')) {
        break
      }
    } catch {
      // The detached child has not created the breadcrumb yet.
    }

    await new Promise(resolve => setTimeout(resolve, 25))
  }

  assert.equal(fs.readFileSync(helperLog, 'utf8'), 'nohup\npython3\nbash\n')
})

test.skipIf(process.platform === 'win32')('POSIX hand-off uses setsid without a Python dependency', async () => {
  const { home, installRoot } = sandbox('setsid-no-python')
  const mockBin = path.join(home, 'mock-bin')
  const realBash = executableOnPath('bash')
  const realSh = executableOnPath('sh')
  const helperLog = path.join(home, 'helpers.log')

  fs.mkdirSync(mockBin)

  for (const command of ['cat', 'date', 'dirname', 'mkdir', 'pwd', 'rm', 'tee']) {
    fs.symlinkSync(executableOnPath(command), path.join(mockBin, command))
  }

  fs.writeFileSync(
    path.join(mockBin, 'bash'),
    `#!${realSh}\nprintf 'bash\\n' >> ${JSON.stringify(helperLog)}\nexec ${JSON.stringify(realBash)} "$@"\n`,
    { mode: 0o755 }
  )
  fs.writeFileSync(
    path.join(mockBin, 'setsid'),
    `#!${realSh}\nprintf 'setsid\\n' >> ${JSON.stringify(helperLog)}\nexec "$@"\n`,
    { mode: 0o755 }
  )

  const result = spawnSync(realBash, [POSIX_SCRIPT, '--install-root', installRoot, '--self-test-marker'], {
    env: { ...process.env, PATH: mockBin },
    encoding: 'utf8'
  })

  assert.equal(result.status, 0, String(result.stderr || result.stdout))

  for (let attempt = 0; attempt < 40; attempt += 1) {
    try {
      if (fs.readFileSync(helperLog, 'utf8').includes('bash\n')) {
        break
      }
    } catch {
      // The detached child has not created the breadcrumb yet.
    }

    await new Promise(resolve => setTimeout(resolve, 25))
  }

  assert.equal(fs.readFileSync(helperLog, 'utf8'), 'setsid\nbash\n')
})

function runWindows(installRoot: string, startedAt?: string) {
  const env = { ...process.env }

  if (startedAt === undefined) {
    delete env.HERMES_UPDATE_STARTED_AT
  } else {
    env.HERMES_UPDATE_STARTED_AT = startedAt
  }

  return spawnSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-ExecutionPolicy',
      'Bypass',
      '-File',
      WINDOWS_SCRIPT,
      '-InstallRoot',
      installRoot,
      '-NoUi',
      '-NoMarkerCleanup',
      '-SelfTestMarker'
    ],
    { env, encoding: 'utf8' }
  )
}

function assertScriptHandoff(run: (installRoot: string, startedAt?: string) => ReturnType<typeof spawnSync>) {
  const preserved = sandbox('preserved')
  const acquiredAt = Math.floor(Date.now() / 1000) - 300
  const preservedResult = run(preserved.installRoot, String(acquiredAt))

  assert.equal(preservedResult.status, 0, String(preservedResult.stderr || preservedResult.stdout))
  assert.equal(markerStartedAt(preserved.home), acquiredAt, 'the script must preserve the Desktop acquisition time')

  const refreshed = sandbox('refreshed')
  fs.writeFileSync(path.join(refreshed.home, '.hermes-update-in-progress'), '999999\n1\n')
  const before = Math.floor(Date.now() / 1000)
  const refreshedResult = run(refreshed.installRoot, 'malformed')
  const after = Math.floor(Date.now() / 1000)

  assert.equal(refreshedResult.status, 0, String(refreshedResult.stderr || refreshedResult.stdout))
  assert.ok(
    markerStartedAt(refreshed.home) >= before && markerStartedAt(refreshed.home) <= after,
    'an invalid hand-off timestamp must start a fresh claim'
  )

  const oversized = sandbox('oversized')
  const oversizedBefore = Math.floor(Date.now() / 1000)
  const oversizedResult = run(oversized.installRoot, '99999999999999999999')
  const oversizedAfter = Math.floor(Date.now() / 1000)

  assert.equal(oversizedResult.status, 0, String(oversizedResult.stderr || oversizedResult.stdout))
  assert.ok(
    markerStartedAt(oversized.home) >= oversizedBefore && markerStartedAt(oversized.home) <= oversizedAfter,
    'an oversized hand-off timestamp must start a fresh claim'
  )
}

test.skipIf(process.platform === 'win32')('POSIX hand-off preserves the Desktop marker acquisition time', () => {
  assertScriptHandoff(runPosix)
})

test.skipIf(process.platform !== 'win32')('PowerShell hand-off preserves the Desktop marker acquisition time', () => {
  assertScriptHandoff(runWindows)
})
