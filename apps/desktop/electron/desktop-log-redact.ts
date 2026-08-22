import { redactSecrets } from './ssh-connection'

const MAX_CARRY_CHARS = 4096

const SECRET_PREFIX_RE =
  /(?:[?&](?:token|ticket)=|HERMES_DASHBOARD_SESSION_TOKEN=|X-Hermes-Session-Token["']?\s*[:=]\s*["']?|Authorization["']?\s*:\s*Bearer\s+)/i

export function formatDesktopLogChunk(chunk: unknown): string {
  return redactSecrets(String(chunk ?? '')).trim()
}

export function createDesktopLogFormatter(): (chunk: unknown) => string {
  let carry = ''
  let suppressSecret = false

  return chunk => {
    let input = String(chunk ?? '')

    if (suppressSecret) {
      const boundary = input.search(/\r?\n/)

      if (boundary < 0) {
        return ''
      }

      input = input.slice(boundary + (input[boundary] === '\r' ? 2 : 1))
      suppressSecret = false
    }

    carry += input
    const parts = carry.split(/\r?\n/)
    carry = parts.pop() ?? ''
    let complete = parts.join('\n')

    if (carry && !SECRET_PREFIX_RE.test(carry)) {
      complete += `${complete ? '\n' : ''}${carry}`
      carry = ''
    }

    if (carry.length > MAX_CARRY_CHARS && SECRET_PREFIX_RE.test(carry)) {
      carry = ''
      suppressSecret = true
    }

    return formatDesktopLogChunk(complete)
  }
}
