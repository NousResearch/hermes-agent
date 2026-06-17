const IPC_REMOTE_ERROR_RE = /^Error invoking remote method '[^']+': Error: (.+)$/s
const HTTP_ERROR_RE = /^\d{3}:\s*(.+)$/s

function rawErrorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error) {
    return error.message
  }

  if (typeof error === 'string') {
    return error
  }

  return fallback
}

function cleanErrorText(value: string): string {
  return value.replace(/^Error:\s*/, '').trim()
}

function detailToString(detail: unknown): null | string {
  if (typeof detail === 'string') {
    return detail
  }

  if (Array.isArray(detail)) {
    const messages = detail
      .map(item => {
        if (typeof item === 'string') {
          return item
        }

        if (item && typeof item === 'object' && 'msg' in item && typeof item.msg === 'string') {
          return item.msg
        }

        return null
      })
      .filter((item): item is string => Boolean(item))

    return messages.length > 0 ? messages.join('\n') : null
  }

  return null
}

function parseJsonDetail(value: string): null | string {
  try {
    const parsed = JSON.parse(value) as unknown

    if (parsed && typeof parsed === 'object' && 'detail' in parsed) {
      return detailToString(parsed.detail)
    }
  } catch {
    return null
  }

  return null
}

export function formatBackendError(error: unknown, fallback: string): string {
  const raw = rawErrorMessage(error, fallback)
  const withoutIpc = raw.match(IPC_REMOTE_ERROR_RE)?.[1] ?? raw
  const cleaned = cleanErrorText(withoutIpc)
  const payload = cleaned.match(HTTP_ERROR_RE)?.[1] ?? cleaned
  const detail = parseJsonDetail(payload)
  const formatted = cleanErrorText(detail ?? cleaned)

  return formatted || fallback
}
