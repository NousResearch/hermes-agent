const GENERIC_PATH_SEGMENTS = new Set(['app', 'apps', 'desktop', 'home', 'src', 'users', 'workspace'])
// xAI caps each transcription keyterm at 50 characters (mirrors tools/voice_realtime_config.py).
export const MAX_KEYTERM_CHARS = 50

export function realtimeVoiceContextKeyterms(cwd?: null | string): string[] {
  if (!cwd) {
    return []
  }

  const terms: string[] = []
  const seen = new Set<string>()

  for (const segment of cwd.split(/[\\/]/).slice(-3)) {
    const term = segment.trim()
    const key = term.toLowerCase()

    if (term.length < 3 || GENERIC_PATH_SEGMENTS.has(key) || seen.has(key)) {
      continue
    }

    seen.add(key)
    terms.push(term.slice(0, MAX_KEYTERM_CHARS))
  }

  return terms
}
