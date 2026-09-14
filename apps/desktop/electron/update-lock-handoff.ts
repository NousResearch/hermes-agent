import { execFileSync } from 'node:child_process'
import path from 'node:path'

import { hiddenWindowsChildOptions } from './windows-child-options'

export interface WindowsProcessEntry {
  commandLine: string
  executablePath: string
  pid: number
}

type ProcessListRunner = (file: string, args: string[], options: unknown) => string

function normalizeWindowsPathForCompare(value: string): string {
  return path.win32.normalize(String(value || '')).replaceAll('/', '\\').toLowerCase()
}

export function isSameInstallVenvHolder(entry: Partial<WindowsProcessEntry>, updateRoot: string): boolean {
  const commandLine = String(entry.commandLine || '')
  const executablePath = String(entry.executablePath || '')
  const commandLineLow = commandLine.toLowerCase().replaceAll('/', '\\')
  const executableLow = executablePath.toLowerCase().replaceAll('/', '\\')
  const normalizedRoot = normalizeWindowsPathForCompare(updateRoot)
  const rootPrefix = `${normalizedRoot}\\`
  const venvPrefix = `${rootPrefix}venv\\`

  if (executableLow.startsWith(venvPrefix)) {
    return true
  }

  if (commandLineLow.includes(venvPrefix)) {
    return true
  }

  return commandLineLow.includes('hermes_cli.main') && commandLineLow.includes(normalizedRoot)
}

export function isGatewayRunCommand(commandLine: string): boolean {
  return String(commandLine || '').toLowerCase().includes('gateway run')
}

export function canHandOffLockedUpdateToInstaller(
  updateRoot: string,
  entries: Array<Partial<WindowsProcessEntry>>
): boolean {
  const holders = entries.filter(entry => isSameInstallVenvHolder(entry, updateRoot))

  return holders.length > 0 && holders.every(entry => isGatewayRunCommand(String(entry.commandLine || '')))
}

export function parseWindowsProcessList(text: string): WindowsProcessEntry[] {
  const entries: WindowsProcessEntry[] = []
  let current: Partial<WindowsProcessEntry> = {}

  const flush = () => {
    if (Number.isInteger(current.pid) && current.pid! > 0) {
      entries.push({
        commandLine: String(current.commandLine || ''),
        executablePath: String(current.executablePath || ''),
        pid: Number(current.pid)
      })
    }
    current = {}
  }

  for (const rawLine of String(text || '').split(/\r?\n/)) {
    const line = rawLine.trim()

    if (!line) {
      flush()
      continue
    }

    if (line.startsWith('CommandLine=')) {
      current.commandLine = line.slice('CommandLine='.length)
      continue
    }

    if (line.startsWith('ExecutablePath=')) {
      current.executablePath = line.slice('ExecutablePath='.length)
      continue
    }

    if (line.startsWith('ProcessId=')) {
      const pid = Number.parseInt(line.slice('ProcessId='.length), 10)

      if (Number.isInteger(pid)) {
        current.pid = pid
      }
    }
  }

  flush()
  return entries
}

export function listWindowsInstallVenvHolders(
  updateRoot: string,
  run: ProcessListRunner = (file, args, options) => execFileSync(file, args, options as any)
): WindowsProcessEntry[] {
  const commands: Array<[string, string[]]> = [
    ['wmic', ['process', 'get', 'ProcessId,ExecutablePath,CommandLine', '/FORMAT:LIST']],
    [
      'powershell',
      [
        '-NoProfile',
        '-Command',
        "Get-CimInstance Win32_Process | ForEach-Object { 'CommandLine=' + ($_.CommandLine -replace \"`r`n\",' ' -replace \"`n\",' '); 'ExecutablePath=' + $_.ExecutablePath; 'ProcessId=' + $_.ProcessId; '' }"
      ]
    ],
    [
      'pwsh',
      [
        '-NoProfile',
        '-Command',
        "Get-CimInstance Win32_Process | ForEach-Object { 'CommandLine=' + ($_.CommandLine -replace \"`r`n\",' ' -replace \"`n\",' '); 'ExecutablePath=' + $_.ExecutablePath; 'ProcessId=' + $_.ProcessId; '' }"
      ]
    ]
  ]

  for (const [command, args] of commands) {
    try {
      const stdout = run(
        command,
        args,
        hiddenWindowsChildOptions({
          encoding: 'utf8',
          stdio: ['ignore', 'pipe', 'ignore']
        })
      )

      return parseWindowsProcessList(stdout).filter(
        entry => entry.pid !== process.pid && isSameInstallVenvHolder(entry, updateRoot)
      )
    } catch {
      void 0
    }
  }

  return []
}
