export type FormatJsonResult = { ok: true; text: string } | { ok: false; error: string }

export function tryFormatJson(raw: string): FormatJsonResult {
  const text = raw.trim()

  if (!text) {
    return { ok: true, text: raw }
  }

  try {
    return { ok: true, text: JSON.stringify(JSON.parse(text) as unknown, null, 2) }
  } catch (err) {
    return { ok: false, error: err instanceof Error ? err.message : String(err) }
  }
}
