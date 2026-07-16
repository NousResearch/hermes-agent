import { execFile, execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

const execFileAsync = promisify(execFile)

let cachedDistributions: string[] | null = null

export const TERMINAL_MODES = ['smart', 'wsl2', 'windows-native'] as const

export type TerminalMode = (typeof TERMINAL_MODES)[number]

interface ResolveTerminalModeOptions {
  configuredMode: TerminalMode
  cwd?: string
  isWindows: boolean
  wslDistributions: string[]
}

export function normalizeTerminalMode(value: unknown): TerminalMode {
  return TERMINAL_MODES.includes(value as TerminalMode) ? (value as TerminalMode) : 'smart'
}

export function terminalModeConfigPath(hermesHome: string): string {
  return path.join(hermesHome, 'config.yaml')
}

function stripYamlScalar(value: string): string {
  const withoutComment = value.replace(/\s+#.*$/, '').trim()
  const quote = withoutComment[0]

  if ((quote === '"' || quote === "'") && withoutComment.endsWith(quote)) {
    return withoutComment.slice(1, -1)
  }

  return withoutComment
}

export function readTerminalMode(hermesHome: string): TerminalMode {
  try {
    const lines = fs.readFileSync(terminalModeConfigPath(hermesHome), 'utf8').split(/\r?\n/)
    let inTerminal = false

    for (const line of lines) {
      if (!/^\s/.test(line)) {
        inTerminal = /^terminal\s*:\s*(?:#.*)?$/.test(line)
        continue
      }

      if (inTerminal) {
        const match = /^\s+windows_execution_mode\s*:\s*(.*?)\s*$/.exec(line)
        if (match) {
          return normalizeTerminalMode(stripYamlScalar(match[1]))
        }
      }
    }
  } catch {
    // Missing or unreadable config falls back to the documented default.
  }

  return 'smart'
}

export function writeTerminalMode(hermesHome: string, value: unknown): TerminalMode {
  const mode = normalizeTerminalMode(value)
  const target = terminalModeConfigPath(hermesHome)
  fs.mkdirSync(path.dirname(target), { recursive: true })

  let text = ''
  try {
    text = fs.readFileSync(target, 'utf8')
  } catch {
    // A fresh install may not have config.yaml yet.
  }

  const lines = text ? text.split(/\r?\n/) : []
  let terminalStart = -1
  let terminalEnd = lines.length
  let modeLine = -1

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index]
    if (!/^\s/.test(line)) {
      if (/^terminal\s*:\s*(?:#.*)?$/.test(line)) {
        terminalStart = index
        terminalEnd = lines.length
      } else if (terminalStart >= 0) {
        terminalEnd = index
        break
      }
      continue
    }

    if (terminalStart >= 0 && /^\s+windows_execution_mode\s*:/.test(line)) {
      modeLine = index
    }
  }

  if (modeLine >= 0) {
    const indent = /^\s*/.exec(lines[modeLine])?.[0] || '  '
    lines[modeLine] = `${indent}windows_execution_mode: ${mode}`
  } else if (terminalStart >= 0) {
    lines.splice(terminalEnd, 0, `  windows_execution_mode: ${mode}`)
  } else {
    while (lines.length && lines[lines.length - 1] === '') {
      lines.pop()
    }
    if (lines.length) {
      lines.push('')
    }
    lines.push('terminal:', `  windows_execution_mode: ${mode}`)
  }

  fs.writeFileSync(target, `${lines.join('\n').replace(/\n+$/, '')}\n`, 'utf8')
  return mode
}

export function parseWslDistributions(output: unknown): string[] {
  let text = ''
  if (Buffer.isBuffer(output)) {
    text = output.toString('utf16le')
  } else {
    text = String(output || '')
  }
  const seen = new Set<string>()

  return text
    .split('\u0000')
    .join('')
    .split(/\r?\n/)
    .map(line => line.replace(/^\*\s*/, '').trim())
    .filter(name => {
      const key = name.toLowerCase()

      if (!name || seen.has(key)) {
        return false
      }

      seen.add(key)

      return true
    })
}

export function detectWslDistributions(run: typeof execFileSync = execFileSync): string[] {
  try {
    const output = run('wsl.exe', ['--list', '--quiet'], {
      encoding: 'buffer' as any,
      stdio: ['ignore', 'pipe', 'ignore'],
      windowsHide: true
    })

    return parseWslDistributions(output)
  } catch {
    return []
  }
}

export async function detectWslDistributionsAsync(): Promise<string[]> {
  if (cachedDistributions !== null) {
    return cachedDistributions
  }
  try {
    const { stdout } = await execFileAsync('wsl.exe', ['--list', '--quiet'], {
      encoding: 'buffer',
      windowsHide: true
    })
    cachedDistributions = parseWslDistributions(stdout)
    return cachedDistributions
  } catch {
    cachedDistributions = []
    return []
  }
}

export function clearWslDistributionCache(): void {
  cachedDistributions = null
}

export function distributionFromWslPath(cwd: string): string | null {
  const match = /^\\\\(?:wsl\.localhost|wsl\$)\\([^\\/]+)(?:[\\/]|$)/i.exec(String(cwd || ''))

  return match?.[1] || null
}

function matchingDistribution(distributions: string[], requested: string | null): string | null {
  if (!requested) {
    return null
  }

  return distributions.find(name => name.toLowerCase() === requested.toLowerCase()) || null
}

export function resolveWindowsTerminalMode({
  configuredMode,
  cwd = '',
  isWindows,
  wslDistributions
}: ResolveTerminalModeOptions): { distribution: string | null; mode: 'windows-native' | 'wsl2' } {
  if (!isWindows || configuredMode === 'windows-native') {
    return { distribution: null, mode: 'windows-native' }
  }

  const pathDistribution = matchingDistribution(wslDistributions, distributionFromWslPath(cwd))

  if (configuredMode === 'smart') {
    return pathDistribution
      ? { distribution: pathDistribution, mode: 'wsl2' }
      : { distribution: null, mode: 'windows-native' }
  }

  const distribution = pathDistribution || wslDistributions[0] || null

  return { distribution, mode: 'wsl2' }
}

export function toWslPath(cwd: string, distribution?: string | null): string {
  const value = String(cwd || '').trim()

  if (!value) {
    return '~'
  }

  const unc = /^\\\\(?:wsl\.localhost|wsl\$)\\([^\\/]+)(?:[\\/](.*))?$/i.exec(value)

  if (unc) {
    if (!distribution || unc[1].toLowerCase() === distribution.toLowerCase()) {
      return (
        `/${String(unc[2] || '')
          .replace(/\\/g, '/')
          .replace(/^\/+/, '')}`.replace(/\/$/, '') || '/'
      )
    }
  }

  const drive = /^([a-zA-Z]):[\\/]*(.*)$/.exec(value)

  if (drive) {
    const tail = drive[2].replace(/\\/g, '/').replace(/^\/+/, '')

    return `/mnt/${drive[1].toLowerCase()}${tail ? `/${tail}` : ''}`
  }

  return value.replace(/\\/g, '/')
}
