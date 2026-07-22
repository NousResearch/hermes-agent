/**
 * windows-update-launch.ts
 *
 * Launch the staged Windows updater OUTSIDE the desktop's Job Object.
 *
 * Why: the desktop process usually lives inside a Windows Job Object with
 * KILL_ON_JOB_CLOSE (scheduled-task launchers, the Tauri bootstrap chain).
 * Node's `detached: true` does NOT set CREATE_BREAKAWAY_FROM_JOB, so a
 * directly spawned hermes-setup.exe stays in that job and dies together with
 * the desktop when it quits for the hand-off — bootstrap-installer.log then
 * shows only the init line and the update never runs. Processes started by
 * the Task Scheduler service are not part of the desktop's job, so a
 * one-shot task is used as the launch vehicle.
 *
 * The launcher script writes an ACK file once the updater process is running;
 * the desktop waits for that ACK before quitting, turning the hand-off from
 * fire-and-forget into a positively confirmed exchange. Ambiguous launch
 * status fails closed; callers must never start a second updater in parallel.
 */

import { execFile } from 'node:child_process'
import crypto from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { hiddenWindowsChildOptions } from './windows-child-options'

export const UPDATE_LAUNCH_TASK_NAME = 'Hermes_UpdateLaunch'

export type UpdateLaunchPaths = {
  scriptPath: string
  ackPath: string
  logPath: string
}

export function updateLaunchPaths(hermesHome: string, requestId = ''): UpdateLaunchPaths {
  const logsDir = path.join(hermesHome, 'logs')
  const suffix = requestId ? `-${requestId.replace(/[^a-z0-9]/gi, '')}` : ''

  return {
    scriptPath: path.join(hermesHome, `update-launch${suffix}.ps1`),
    ackPath: path.join(logsDir, `update-launch-ack${suffix}.json`),
    logPath: path.join(logsDir, 'update-launch.log')
  }
}

// Single-quoted PowerShell string literal: only ' needs doubling.
function psQuote(value: string): string {
  return `'${String(value).replace(/'/g, "''")}'`
}

export type LaunchScriptArgs = {
  updaterPath: string
  updaterArgs: string[]
  hermesHome: string
  pathEnv: string
  ackPath: string
  logPath: string
  requestId: string
  taskName?: string
}

/**
 * Build the PowerShell launcher the one-shot task executes. It runs outside
 * the desktop's job, starts the updater, writes the ACK (with the updater
 * PID), then stays alive until the updater exits so Task Scheduler's
 * IgnoreNew instance policy shields against duplicate firings, and finally
 * removes the one-shot task definition.
 */
export function buildUpdateLaunchScript(args: LaunchScriptArgs): string {
  const taskName = args.taskName || UPDATE_LAUNCH_TASK_NAME
  const updaterArgsList = args.updaterArgs.map(psQuote).join(', ')

  return [
    `$ErrorActionPreference = 'Stop'`,
    `$ack = ${psQuote(args.ackPath)}`,
    `$log = ${psQuote(args.logPath)}`,
    `$requestId = ${psQuote(args.requestId)}`,
    `function Write-LaunchLog([string]$m) {`,
    `  try { Add-Content -Path $log -Value ("{0} [{1}] {2}" -f (Get-Date).ToString('o'), $requestId, $m) } catch {}`,
    `}`,
    `# Duplicate-invocation guard: the schedule (a past-tense one-shot) should`,
    `# never fire on its own, but if it ever does, the ACK for this request id`,
    `# marks the work as already claimed.`,
    `if (Test-Path -LiteralPath $ack) {`,
    `  $existing = $null`,
    `  try { $existing = Get-Content -LiteralPath $ack -Raw | ConvertFrom-Json } catch {}`,
    `  if ($existing -and $existing.requestId -eq $requestId) {`,
    `    Write-LaunchLog 'duplicate invocation ignored'`,
    `    exit 0`,
    `  }`,
    `}`,
    `try {`,
    `  $env:HERMES_HOME = ${psQuote(args.hermesHome)}`,
    `  $env:PATH = ${psQuote(args.pathEnv)}`,
    `  Write-LaunchLog 'launcher started (outside desktop job object)'`,
    `  $p = Start-Process -FilePath ${psQuote(args.updaterPath)} -ArgumentList @(${updaterArgsList}) -WorkingDirectory $env:HERMES_HOME -PassThru`,
    `  $payload = @{ ok = $true; requestId = $requestId; updaterPid = $p.Id; startedAt = (Get-Date).ToString('o') } | ConvertTo-Json -Compress`,
    `  Set-Content -LiteralPath $ack -Value $payload -Encoding UTF8`,
    `  Write-LaunchLog ("updater started pid=" + $p.Id)`,
    `  try { Wait-Process -Id $p.Id -ErrorAction SilentlyContinue } catch {}`,
    `  Write-LaunchLog 'updater exited; removing one-shot task'`,
    `} catch {`,
    `  $err = @{ ok = $false; requestId = $requestId; error = $_.Exception.Message } | ConvertTo-Json -Compress`,
    `  try { Set-Content -LiteralPath $ack -Value $err -Encoding UTF8 } catch {}`,
    `  Write-LaunchLog ("launcher failed: " + $_.Exception.Message)`,
    `} finally {`,
    `  try { schtasks.exe /Delete /F /TN ${psQuote(taskName)} | Out-Null } catch {}`,
    `}`,
    ``
  ].join('\r\n')
}

/**
 * Next /ST value (HH:mm, locale-independent) a couple of minutes ahead, so
 * schtasks never warns about a start time in the past. The schedule itself is
 * irrelevant — the task is triggered explicitly via /Run and the launcher's
 * request-id guard ignores any stray timed firing.
 */
export function nextStartTime(now: Date = new Date(), minutesAhead = 2): string {
  const t = new Date(now.getTime() + minutesAhead * 60_000)
  const hh = String(t.getHours()).padStart(2, '0')
  const mm = String(t.getMinutes()).padStart(2, '0')

  return `${hh}:${mm}`
}

export function schtasksCreateArgs(taskName: string, scriptPath: string, startTime: string): string[] {
  const action = `powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "${scriptPath}"`

  return ['/Create', '/F', '/TN', taskName, '/SC', 'ONCE', '/ST', startTime, '/TR', action]
}

export function schtasksRunArgs(taskName: string): string[] {
  return ['/Run', '/TN', taskName]
}

export function schtasksEndArgs(taskName: string): string[] {
  return ['/End', '/TN', taskName]
}

export function schtasksDeleteArgs(taskName: string): string[] {
  return ['/Delete', '/F', '/TN', taskName]
}

export type LaunchAck = {
  ok: boolean
  requestId: string
  updaterPid: number | null
  error: string | null
}

export function parseLaunchAck(raw: unknown, requestId: string): LaunchAck | null {
  const text = String(raw ?? '')
    .replace(/^\uFEFF/, '')
    .trim()

  if (!text) {
    return null
  }

  let parsed: any

  try {
    parsed = JSON.parse(text)
  } catch {
    return null
  }

  if (!parsed || parsed.requestId !== requestId) {
    return null
  }

  const pid = Number(parsed.updaterPid)

  return {
    ok: parsed.ok === true,
    requestId,
    updaterPid: Number.isInteger(pid) && pid > 0 ? pid : null,
    error: typeof parsed.error === 'string' ? parsed.error : null
  }
}

export type RunCommand = (file: string, args: string[]) => Promise<{ code: number; stdout: string; stderr: string }>

function defaultRunCommand(file: string, args: string[]): Promise<{ code: number; stdout: string; stderr: string }> {
  return new Promise(resolve => {
    execFile(
      file,
      args,
      hiddenWindowsChildOptions({ encoding: 'utf8' }),
      (error: any, stdout: any, stderr: any) => {
        resolve({
          code: error ? (typeof error.code === 'number' ? error.code : 1) : 0,
          stdout: String(stdout ?? ''),
          stderr: String(stderr ?? '')
        })
      }
    )
  })
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

export type LaunchResult = { ok: boolean; updaterPid: number | null; error: string | null }

function launchFailure(error: string): LaunchResult {
  return { ok: false, updaterPid: null, error }
}

/**
 * Full launch flow: write the launcher script, register + fire the one-shot
 * task, then wait for the launcher's ACK. Never throws — callers use the
 * error result to keep the desktop alive and report the failed handoff.
 */
export async function launchUpdaterViaScheduledTask(opts: {
  updaterPath: string
  updaterArgs: string[]
  hermesHome: string
  pathEnv: string
  taskName?: string
  runCommand?: RunCommand
  timeoutMs?: number
  pollMs?: number
  now?: Date
}): Promise<LaunchResult> {
  const runCommand = opts.runCommand || defaultRunCommand
  const timeoutMs = opts.timeoutMs ?? 30_000
  const pollMs = opts.pollMs ?? 250
  const requestId = crypto.randomUUID()
  const taskName = opts.taskName || `${UPDATE_LAUNCH_TASK_NAME}_${requestId.replaceAll('-', '')}`
  const paths = updateLaunchPaths(opts.hermesHome, requestId)

  const cleanupFiles = (): void => {
    try {
      fs.rmSync(paths.scriptPath, { force: true })
      fs.rmSync(paths.ackPath, { force: true })
    } catch {
      // Best effort; request-specific filenames make leftovers inert.
    }
  }

  const cleanupTask = async (endRunning: boolean): Promise<void> => {
    if (endRunning) {
      try {
        await runCommand('schtasks.exe', schtasksEndArgs(taskName))
      } catch {
        // Best effort. Deleting the definition below is still required even
        // when Task Scheduler cannot confirm that the action has ended.
      }
    }

    try {
      await runCommand('schtasks.exe', schtasksDeleteArgs(taskName))
    } catch {
      // Best effort. The unique task name prevents collision with a later
      // request even if Task Scheduler is temporarily unavailable.
    }
  }

  try {
    fs.mkdirSync(path.dirname(paths.ackPath), { recursive: true })
    fs.rmSync(paths.ackPath, { force: true })
    fs.writeFileSync(
      paths.scriptPath,
      buildUpdateLaunchScript({
        updaterPath: opts.updaterPath,
        updaterArgs: opts.updaterArgs,
        hermesHome: opts.hermesHome,
        pathEnv: opts.pathEnv,
        ackPath: paths.ackPath,
        logPath: paths.logPath,
        requestId,
        taskName
      }),
      'utf8'
    )
  } catch (err: any) {
    return launchFailure(`launcher script write failed: ${err.message}`)
  }

  const create = await runCommand('schtasks.exe', schtasksCreateArgs(taskName, paths.scriptPath, nextStartTime(opts.now)))

  if (create.code !== 0) {
    await cleanupTask(false)
    cleanupFiles()

    return launchFailure(`schtasks create failed (${create.code}): ${(create.stderr || create.stdout).trim()}`)
  }

  const run = await runCommand('schtasks.exe', schtasksRunArgs(taskName))

  if (run.code !== 0) {
    await cleanupTask(true)
    cleanupFiles()

    return launchFailure(`schtasks run failed (${run.code}): ${(run.stderr || run.stdout).trim()}`)
  }

  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    let raw: string | null = null

    try {
      raw = fs.readFileSync(paths.ackPath, 'utf8')
    } catch {
      // ACK not written yet.
    }

    const ack = raw === null ? null : parseLaunchAck(raw, requestId)

    if (ack) {
      if (ack.ok && ack.updaterPid !== null) {
        cleanupFiles()

        return { ok: true, updaterPid: ack.updaterPid, error: null }
      }

      await cleanupTask(true)
      cleanupFiles()

      return launchFailure(`launcher reported failure: ${ack.error || 'missing updater pid'}`)
    }

    await sleep(pollMs)
  }

  // Remove the scheduled definition so it cannot fire at its nominal time
  // after /Run failed or was delayed. Never start a fallback updater here: if
  // the task already started but its ACK was inaccessible, the live desktop
  // makes it fail on the holder guard instead of racing a second mutator.
  await cleanupTask(true)
  cleanupFiles()

  return launchFailure(`no launcher ACK within ${timeoutMs}ms`)
}
