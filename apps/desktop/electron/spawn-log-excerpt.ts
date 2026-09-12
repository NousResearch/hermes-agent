const ANSI_RE =
  /\u001B\[[0-9;?]*[ -/]*[@-~]|\u001B\][^\u0007\u001B]*(?:\u0007|\u001B\\)|\u001B[PX^_][^\u001B]*\u001B\\|\u001B[@-Z\\-_]/g
const MAX_LINES = 20
const MAX_CHARS = 2000

const SECRET_LINE_RE =
  /-----BEGIN [A-Z ]*PRIVATE KEY-----|HERMES_DASHBOARD_SESSION_TOKEN=|(?:^|[\s])(?:token|api[_-]?key|secret|password|passwd|authorization|identityfile)\s*[=:]|(?:^|[\s])Bearer\s+\S+|sk-[A-Za-z0-9]{10,}|\.ssh\/id_[A-Za-z0-9_-]+/i

const LOCAL_SPAWN_LOG_RE = /(?:~|\/home\/|\/Users\/|[A-Za-z]:\\)[^\s]*desktop-ssh[^\s]*\.(?:log|txt)/i

export function excerptSpawnLog(raw: unknown): string {
  if (raw == null) {
    return ''
  }

  const lines = String(raw)
    .replace(ANSI_RE, '')
    .split(/\r?\n/)
    .map(line => line.trimEnd())
    .filter(line => {
      const trimmed = line.trim()

      if (!trimmed) {
        return false
      }

      if (SECRET_LINE_RE.test(trimmed) || LOCAL_SPAWN_LOG_RE.test(trimmed)) {
        return false
      }

      if (/^\s*cat\s+\S+\.log\b/.test(trimmed)) {
        return false
      }

      return true
    })

  const excerpt = lines.slice(-MAX_LINES).join('\n').trim()

  if (!excerpt) {
    return ''
  }

  return excerpt.length > MAX_CHARS ? excerpt.slice(-MAX_CHARS) : excerpt
}

export function attachSpawnLogDetail<T extends { detail?: string }>(err: T, raw: unknown): T {
  const detail = excerptSpawnLog(raw)

  if (detail) {
    err.detail = detail
  }

  return err
}
