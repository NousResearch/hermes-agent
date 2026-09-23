import { execFileSync } from 'node:child_process'

interface PythonSelection {
  path: string
  version: string
}

type ReadPythonVersion = (candidate: string) => string | null
type FindOnPath = (command: string) => string | null
type ExecVersionProbe = (
  command: string,
  args: string[],
  options: {
    encoding: 'utf8'
    stdio: ['ignore', 'pipe', 'ignore']
    timeout: number
    windowsHide: boolean
  }
) => string | Buffer

const PYTHON_VERSION_SNIPPET = 'import sys; print(".".join(map(str, sys.version_info[:3])))'
const PYTHON_VERSION_PROBE_TIMEOUT_MS = 5_000

function readPythonVersion(
  candidate: string,
  exec: ExecVersionProbe = execFileSync as ExecVersionProbe
): string | null {
  try {
    const output = exec(candidate, ['-c', PYTHON_VERSION_SNIPPET], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      timeout: PYTHON_VERSION_PROBE_TIMEOUT_MS,
      windowsHide: true
    })

    const version = String(output).trim()

    return version || null
  } catch {
    return null
  }
}

function isSupportedPythonVersion(version: string): boolean {
  const match = String(version).trim().match(/^(\d+)\.(\d+)(?:\.\d+)?$/)

  if (!match) {
    return false
  }

  const major = Number(match[1])
  const minor = Number(match[2])

  return major === 3 && minor >= 11 && minor < 14
}

function selectSupportedPythonCandidate(
  candidates: string[],
  readVersion: ReadPythonVersion
): PythonSelection | null {
  for (const candidate of candidates) {
    const version = readVersion(candidate)

    if (version && isSupportedPythonVersion(version)) {
      return { path: candidate, version }
    }
  }

  return null
}

function findSupportedPythonOnPath(
  commands: string[],
  findOnPath: FindOnPath,
  readVersion: ReadPythonVersion = readPythonVersion
): PythonSelection | null {
  const candidates = commands.map(findOnPath).filter((candidate): candidate is string => Boolean(candidate))

  return selectSupportedPythonCandidate(candidates, readVersion)
}

export {
  findSupportedPythonOnPath,
  isSupportedPythonVersion,
  readPythonVersion,
  selectSupportedPythonCandidate
}
export type { ExecVersionProbe, FindOnPath, PythonSelection, ReadPythonVersion }
