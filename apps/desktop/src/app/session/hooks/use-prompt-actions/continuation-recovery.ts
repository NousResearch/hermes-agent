import { readKey, writeKey } from '@/lib/storage'

const continuationPrompts = new Map<string, string>()
const CONTINUATION_PROMPTS_STORAGE_KEY = 'hermes.desktop.continuationPrompts.v1'
const MAX_PERSISTED_PROMPTS = 50
const MAX_PROMPT_AGE_MS = 7 * 24 * 60 * 60 * 1000

interface PersistedContinuationPrompt {
  prompt: string
  updatedAt: number
}

function loadPersistedPrompts(now = Date.now()): Record<string, PersistedContinuationPrompt> {
  const raw = readKey(CONTINUATION_PROMPTS_STORAGE_KEY)

  if (!raw) {
    return {}
  }

  try {
    const parsed = JSON.parse(raw)

    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return {}
    }

    return Object.fromEntries(
      Object.entries(parsed).filter((entry): entry is [string, PersistedContinuationPrompt] => {
        const [sessionId, value] = entry

        if (!sessionId || !value || typeof value !== 'object' || Array.isArray(value)) {
          return false
        }

        const record = value as Record<string, unknown>

        return (
          typeof record.prompt === 'string' &&
          record.prompt.trim().length > 0 &&
          typeof record.updatedAt === 'number' &&
          Number.isFinite(record.updatedAt) &&
          now - record.updatedAt <= MAX_PROMPT_AGE_MS
        )
      })
    )
  } catch {
    return {}
  }
}

function persistPrompts(prompts: Record<string, PersistedContinuationPrompt>) {
  const entries = Object.entries(prompts)
    .sort((left, right) => right[1].updatedAt - left[1].updatedAt)
    .slice(0, MAX_PERSISTED_PROMPTS)

  writeKey(CONTINUATION_PROMPTS_STORAGE_KEY, entries.length ? JSON.stringify(Object.fromEntries(entries)) : null)
}

export function rememberContinuationPrompt(sessionId: string | null | undefined, prompt: string): void {
  const sid = String(sessionId || '').trim()

  if (!sid || !prompt.trim()) {
    return
  }

  continuationPrompts.set(sid, prompt)
  persistPrompts({ ...loadPersistedPrompts(), [sid]: { prompt, updatedAt: Date.now() } })
}

export function consumeContinuationPrompt(sessionId: string | null | undefined): string | null {
  const sid = String(sessionId || '').trim()

  if (!sid) {
    return null
  }

  const persisted = loadPersistedPrompts()
  const prompt = continuationPrompts.get(sid) || persisted[sid]?.prompt || null
  continuationPrompts.delete(sid)
  delete persisted[sid]
  persistPrompts(persisted)

  return prompt
}
