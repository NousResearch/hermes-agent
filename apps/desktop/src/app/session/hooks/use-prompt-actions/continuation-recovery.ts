const continuationPrompts = new Map<string, string>()

export function rememberContinuationPrompt(sessionId: string | null | undefined, prompt: string): void {
  const sid = String(sessionId || '').trim()

  if (!sid || !prompt.trim()) {
    return
  }

  continuationPrompts.set(sid, prompt)
}

export function consumeContinuationPrompt(sessionId: string | null | undefined): string | null {
  const sid = String(sessionId || '').trim()

  if (!sid) {
    return null
  }

  const prompt = continuationPrompts.get(sid) || null
  continuationPrompts.delete(sid)

  return prompt
}
