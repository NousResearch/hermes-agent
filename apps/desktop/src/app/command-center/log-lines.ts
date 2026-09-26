export type CommandCenterLogSeverity = 'CRITICAL' | 'DEBUG' | 'ERROR' | 'INFO' | 'WARNING'

const LEVEL_RE = /\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b/

export function commandCenterLogSeverity(line: string): CommandCenterLogSeverity | null {
  const match = LEVEL_RE.exec(line)

  if (!match) {
    return null
  }

  return match[1] === 'WARN' ? 'WARNING' : (match[1] as CommandCenterLogSeverity)
}
