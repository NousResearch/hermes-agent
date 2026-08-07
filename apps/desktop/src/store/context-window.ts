import { atom } from 'nanostores'

/** Overlay visibility for the context-window dialog (`/ctxwindow`). */
export const $contextWindowOpen = atom(false)

export function setContextWindowOpen(open: boolean): void {
  $contextWindowOpen.set(open)
}

/**
 * The context-window figures the dialog renders, as reported by
 * `GET /api/model/info`. All three come from the backend so the desktop never
 * becomes a second, divergent way to compute a context window:
 * `auto_context_length` is `agent.model_metadata.get_model_context_length()`
 * resolved with the override deliberately ignored, and `effective` is what the
 * agent actually uses.
 */
export interface ContextWindowInfo {
  autoContextLength: number
  configContextLength: number
  effectiveContextLength: number
  model: string
  provider: string
}

/** Whether an explicit override is currently pinned. `0`/absent means auto. */
export function hasContextOverride(info: ContextWindowInfo): boolean {
  return info.configContextLength > 0
}

/**
 * The window the agent will actually use.
 *
 * Mirrors the backend's precedence (explicit override wins, else auto-detect)
 * rather than recomputing anything: `get_model_context_length()` returns the
 * config override verbatim at step 0 of its resolution chain. We prefer the
 * backend's own `effective_context_length` and only fall back to the local
 * derivation if that field is missing from an older backend.
 */
export function effectiveContextLength(info: ContextWindowInfo): number {
  if (info.effectiveContextLength > 0) {
    return info.effectiveContextLength
  }

  return hasContextOverride(info) ? info.configContextLength : info.autoContextLength
}

/**
 * Parse a user-typed context length into a persistable override.
 *
 * Returns `0` for "back to auto-detect" (empty input or an explicit 0) and
 * `null` when the text isn't a usable positive integer, so the caller can
 * refuse the save instead of writing a garbage pin. Accepts grouped digits
 * ("200,000" / "200 000") since that's how the value is displayed.
 */
export function parseContextLengthInput(raw: string): number | null {
  const trimmed = raw.trim()

  if (!trimmed) {
    return 0
  }

  if (!/^\d[\d,\s_]*$/.test(trimmed)) {
    return null
  }

  const parsed = Number.parseInt(trimmed.replace(/[,\s_]/g, ''), 10)

  if (!Number.isFinite(parsed) || parsed < 0) {
    return null
  }

  return parsed
}
