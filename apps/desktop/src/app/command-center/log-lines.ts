export type CommandCenterLogSeverity = 'CRITICAL' | 'DEBUG' | 'ERROR' | 'INFO' | 'WARNING'

const LEVEL_RE = /\b(DEBUG|INFO|WARNING|WARN|ERROR|CRITICAL)\b/
