const path = require('node:path')
const { execFileSync } = require('node:child_process')

const RUNTIME_PROCESS_NAMES = new Set([
  'hermes.exe',
  'node.exe',
  'node',
  'npm.cmd',
  'npm.exe',
  'python.exe',
  'pythonw.exe',
  'python',
  'python3',
  'uv.exe',
  'uvicorn.exe'
])

function normalizePathText(value) {
  return String(value || '')
    .replace(/\//g, '\\')
    .toLowerCase()
}

function basenameLower(filePath, fallback = '') {
  const value = String(filePath || fallback || '').trim()
  if (!value) return ''
  return path.basename(value).toLowerCase()
}

function isUnderPath(candidate, root) {
  const normalizedCandidate = normalizePathText(candidate)
  const normalizedRoot = normalizePathText(root).replace(/\\+$/, '')
  if (!normalizedCandidate || !normalizedRoot) return false
  return normalizedCandidate === normalizedRoot || normalizedCandidate.startsWith(`${normalizedRoot}\\`)
}

function textMentionsPath(text, root) {
  const normalizedText = normalizePathText(text)
  const normalizedRoot = normalizePathText(root).replace(/\\+$/, '')
  if (!normalizedText || !normalizedRoot) return false
  let index = normalizedText.indexOf(normalizedRoot)
  while (index !== -1) {
    const after = normalizedText[index + normalizedRoot.length]
    if (!after || after === '\\' || after === '"' || after === "'" || /\s/.test(after)) return true
    index = normalizedText.indexOf(normalizedRoot, index + normalizedRoot.length)
  }
  return false
}

function commandLooksHermesOwned(commandLine) {
  const command = normalizePathText(commandLine)
  return (
    command.includes('hermes_cli.') ||
    command.includes('\\scripts\\whatsapp-bridge\\')
  )
}

function isHermesRuntimeProcess(processInfo, { currentPid = process.pid, updateRoot } = {}) {
  if (!processInfo || !updateRoot) return false

  const pid = Number(processInfo.pid)
  if (!Number.isInteger(pid) || pid <= 0 || pid === currentPid) return false

  const executablePath = processInfo.executablePath || ''
  const commandLine = processInfo.commandLine || ''
  const imageName = basenameLower(processInfo.name || executablePath)
  const executableName = basenameLower(executablePath)
  const processName = imageName || executableName

  if (!RUNTIME_PROCESS_NAMES.has(processName)) return false

  const venvRoot = path.join(updateRoot, 'venv')
  const executableInInstall = isUnderPath(executablePath, updateRoot)
  const executableInVenv = isUnderPath(executablePath, venvRoot)
  const commandMentionsInstall = textMentionsPath(commandLine, updateRoot)
  const commandIsHermes = commandLooksHermesOwned(commandLine)

  return executableInVenv || executableInInstall || (commandMentionsInstall && commandIsHermes)
}

function collectHermesRuntimeProcessIds(processes, { currentPid = process.pid, updateRoot } = {}) {
  return (Array.isArray(processes) ? processes : [])
    .filter(processInfo => isHermesRuntimeProcess(processInfo, { currentPid, updateRoot }))
    .map(processInfo => Number(processInfo.pid))
    .filter(pid => Number.isInteger(pid) && pid > 0)
}

function parseWindowsProcessList(raw) {
  if (!String(raw || '').trim()) return []

  let parsed
  try {
    parsed = JSON.parse(raw)
  } catch {
    return []
  }

  const rows = Array.isArray(parsed) ? parsed : [parsed]
  return rows
    .filter(Boolean)
    .map(row => ({
      pid: Number(row.ProcessId ?? row.processId ?? row.pid),
      name: String(row.Name ?? row.name ?? ''),
      executablePath: String(row.ExecutablePath ?? row.executablePath ?? ''),
      commandLine: String(row.CommandLine ?? row.commandLine ?? '')
    }))
    .filter(row => Number.isInteger(row.pid) && row.pid > 0)
}

function listWindowsProcesses(options = {}) {
  const { onError } = options
  const script =
    "$ErrorActionPreference = 'Stop'; " +
    'Get-CimInstance Win32_Process | ' +
    'Select-Object ProcessId,Name,ExecutablePath,CommandLine | ' +
    'ConvertTo-Json -Compress'

  try {
    const raw = execFileSync('powershell.exe', ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-Command', script], {
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
      timeout: 5000,
      windowsHide: true
    })
    return parseWindowsProcessList(raw)
  } catch (err) {
    onError?.(err)
    return []
  }
}

function killHermesRuntimeProcessesForUpdate(updateRoot, options = {}) {
  const {
    currentPid = process.pid,
    killTree,
    listProcesses = listWindowsProcesses,
    onKilled,
    onError,
    onListError
  } = options

  let processes
  try {
    processes = listProcesses === listWindowsProcesses ? listProcesses({ onError: onListError }) : listProcesses()
  } catch (err) {
    onListError?.(err)
    return []
  }

  const pids = collectHermesRuntimeProcessIds(processes, { currentPid, updateRoot })
  if (pids.length && typeof killTree !== 'function') {
    throw new Error('killHermesRuntimeProcessesForUpdate requires a killTree callback when runtime processes match')
  }

  for (const pid of pids) {
    try {
      killTree(pid)
      onKilled?.(pid)
    } catch (err) {
      onError?.(pid, err)
    }
  }
  return pids
}

module.exports = {
  collectHermesRuntimeProcessIds,
  isHermesRuntimeProcess,
  killHermesRuntimeProcessesForUpdate,
  listWindowsProcesses,
  parseWindowsProcessList
}
