// The IPC bridge hands a router refusal over as "Error invoking remote method 'hermes:api': Error:
// 409: {"detail":{"code":…,"message":…}}"; the card shows the connector's sentence, not the envelope.
export function routerErrorText(error: unknown): string {
  const raw = error instanceof Error ? error.message : String(error)
  const text = (raw.match(/Error invoking remote method '[^']+': Error: (.+)$/s)?.[1] ?? raw).replace(/^Error:\s*/, '')
  const body = text.match(/^\d{3}:\s*(\{.*\})$/s)?.[1]

  if (!body) {
    return text
  }

  try {
    const detail = (JSON.parse(body) as { detail?: unknown }).detail
    const message = typeof detail === 'string' ? detail : (detail as undefined | { message?: unknown })?.message

    return typeof message === 'string' && message ? message : text
  } catch {
    return text
  }
}
