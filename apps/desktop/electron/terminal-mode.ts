import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import path from 'node:path'

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
  return path.join(hermesHome, 'desktop-terminal.json')
}

export function readTerminalMode(hermesHome: string): TerminalMode {
  const envMode = normalizeTerminalMode(process.env.HERMES_WINDOWS_EXECUTION_MODE)

  if (process.env.HERMES_WINDOWS_EXECUTION_MODE) {
    return envMode
  }

  try {
    const parsed = JSON.parse(fs.readFileSync(terminalModeConfigPath(hermesHome), 'utf8'))

    return normalizeTerminalMode(parsed?.mode)
  } catch {
    return 'smart'
  }
}

export function writeTerminalMode(hermesHome: string, value: unknown): TerminalMode {
  const mode = normalizeTerminalMode(value)
  const target = terminalModeConfigPath(hermesHome)
  fs.mkdirSync(path.dirname(target), { recursive: true })
  fs.writeFileSync(target, `${JSON.stringify({ mode }, null, 2)}\n`, 'utf8')

  return mode
}

export function parseWslDistributions(output: unknown): string[] {
  const text = Buffer.isBuffer(output) ? output.toString('utf8') : String(output || '')
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
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      windowsHide: true
    })

    return parseWslDistributions(output)
  } catch {
    return []
  }
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

  return distribution ? { distribution, mode: 'wsl2' } : { distribution: null, mode: 'windows-native' }
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
