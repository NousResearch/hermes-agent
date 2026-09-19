import { atom, computed, type ReadableAtom } from 'nanostores'

import { $activeSessionId } from './session'

export type ComputerUsePhase = 'idle' | 'drafting' | 'running' | 'completed' | 'error'

export interface ComputerUseActiveState {
  phase: ComputerUsePhase
  toolId?: string
  action?: string
  app?: string
  mode?: string
  element?: number
  coordinate?: [number, number]
  text?: string
  keys?: string
  targetSummary?: string
  startedAt?: number
  completedAt?: number
  durationSeconds?: number
  error?: string
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

export const $computerUseBySession = atom<Record<string, ComputerUseActiveState>>({})

/**
 * Normalizes and cleans application / window identifiers.
 * E.g. "C:\\Windows\\notepad.exe" -> "Notepad", "chrome.exe" -> "Chrome", "desktop" -> "Desktop".
 */
export function cleanAppLabel(rawApp: string | undefined): string {
  if (!rawApp) {return ''}
  let name = rawApp.trim()

  // Extract basename if full filesystem path was passed
  const lastSlash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'))

  if (lastSlash >= 0) {
    name = name.slice(lastSlash + 1)
  }

  // Strip .exe or .app extensions
  name = name.replace(/\.(exe|app)$/i, '')

  if (name.toLowerCase() === 'screen') {return 'Screen'}

  if (name.toLowerCase() === 'desktop') {return 'Desktop'}

  // Capitalize first character if single-word lowercase
  if (name && name[0] === name[0].toLowerCase() && !name.includes(' ')) {
    name = name.charAt(0).toUpperCase() + name.slice(1)
  }

  return name
}

/**
 * Format a human-readable, concise summary of the current computer_use operation.
 * E.g. "Chrome · Click #12", "Notepad · Type \"hello\"", "Screen · SOM Capture"
 */
export function formatComputerUseTarget(state: Partial<ComputerUseActiveState>): string {
  const { action, app, mode, element, coordinate, keys, text } = state

  const appLabel = cleanAppLabel(app)
  let actionLabel = ''

  switch (action) {
    case 'capture':
      actionLabel = mode ? `${mode.toUpperCase()} Capture` : 'Capture'

      break

    case 'click':
      actionLabel =
        element != null
          ? `Click #${element}`
          : coordinate
            ? `Click (${coordinate[0]}, ${coordinate[1]})`
            : 'Click'

      break

    case 'double_click':
      actionLabel =
        element != null
          ? `Double Click #${element}`
          : coordinate
            ? `Double Click (${coordinate[0]}, ${coordinate[1]})`
            : 'Double Click'

      break

    case 'right_click':
      actionLabel =
        element != null
          ? `Right Click #${element}`
          : coordinate
            ? `Right Click (${coordinate[0]}, ${coordinate[1]})`
            : 'Right Click'

      break

    case 'middle_click':
      actionLabel =
        element != null
          ? `Middle Click #${element}`
          : coordinate
            ? `Middle Click (${coordinate[0]}, ${coordinate[1]})`
            : 'Middle Click'

      break

    case 'drag':
      actionLabel = coordinate ? `Drag (${coordinate[0]}, ${coordinate[1]})` : 'Drag'

      break

    case 'scroll':
      actionLabel = 'Scroll'

      break

    case 'type':
      if (text) {
        const preview = text.length > 14 ? `${text.slice(0, 12)}…` : text
        actionLabel = `Type "${preview}"`
      } else {
        actionLabel = 'Type'
      }

      break

    case 'key':
      actionLabel = keys ? `Key (${keys})` : 'Key'

      break

    case 'set_value':
      actionLabel = 'Set Value'

      break

    case 'wait':
      actionLabel = 'Wait'

      break

    case 'list_apps':
      actionLabel = 'List Apps'

      break

    case 'list_windows':
      actionLabel = 'List Windows'

      break

    case 'focus_app':
      actionLabel = 'Focus App'

      break

    default:
      if (action) {
        actionLabel = action.replace(/_/g, ' ')
      }

      break
  }

  if (appLabel && actionLabel) {
    return `${appLabel} · ${actionLabel}`
  }

  if (appLabel) {
    return appLabel
  }

  if (actionLabel) {
    return actionLabel
  }

  return 'Active'
}

/** Active computer_use state for the currently focused/active session */
export const $activeComputerUse = computed(
  [$computerUseBySession, $activeSessionId],
  (sessions, activeId) => sessions[keyFor(activeId)] ?? null
)

const sessionComputerUseCache = new Map<string, ReadableAtom<ComputerUseActiveState | null>>()

/** Reactive computer_use state for a specific session */
export function sessionComputerUse(sessionId: string | null | undefined): ReadableAtom<ComputerUseActiveState | null> {
  const key = keyFor(sessionId)
  let cached = sessionComputerUseCache.get(key)

  if (!cached) {
    cached = computed($computerUseBySession, sessions => sessions[key] ?? null)
    sessionComputerUseCache.set(key, cached)
  }

  return cached
}

export const COMPLETED_DISMISS_DELAY_MS = 2500
export const ERROR_DISMISS_DELAY_MS = 5000

const dismissTimers = new Map<string, ReturnType<typeof setTimeout>>()

function cancelDismiss(sessionId: string): void {
  const existing = dismissTimers.get(sessionId)

  if (existing !== undefined) {
    clearTimeout(existing)
    dismissTimers.delete(sessionId)
  }
}

function scheduleDismiss(sessionId: string, delayMs = COMPLETED_DISMISS_DELAY_MS): void {
  cancelDismiss(sessionId)

  const timer = setTimeout(() => {
    dismissTimers.delete(sessionId)
    clearComputerUseState(sessionId)
  }, delayMs)

  dismissTimers.set(sessionId, timer)
}

export function setComputerUseDrafting(sessionId: string | null | undefined): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  cancelDismiss(key)
  const sessions = $computerUseBySession.get()

  $computerUseBySession.set({
    ...sessions,
    [key]: {
      phase: 'drafting',
      startedAt: Date.now(),
      targetSummary: 'Preparing…'
    }
  })
}

export function setComputerUseRunning(
  sessionId: string | null | undefined,
  details: Partial<ComputerUseActiveState>,
  isProgress = false
): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  const sessions = $computerUseBySession.get()
  const existing = sessions[key]

  // If this is an incremental progress update and the action has already settled
  // (completed or errored), ignore late/out-of-order progress events so the dismissal
  // timer or final status isn't overwritten.
  if (isProgress && existing && (existing.phase === 'completed' || existing.phase === 'error')) {
    return
  }

  cancelDismiss(key)

  const mergedState: ComputerUseActiveState = {
    ...existing,
    ...details,
    phase: 'running',
    startedAt: existing?.startedAt ?? Date.now()
  }

  mergedState.targetSummary = details.targetSummary || formatComputerUseTarget(mergedState)

  $computerUseBySession.set({
    ...sessions,
    [key]: mergedState
  })
}

export function setComputerUseCompleted(
  sessionId: string | null | undefined,
  details?: Partial<ComputerUseActiveState>
): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  cancelDismiss(key)
  const sessions = $computerUseBySession.get()
  const existing = sessions[key]

  if (!existing || existing.phase === 'idle') {
    return
  }

  $computerUseBySession.set({
    ...sessions,
    [key]: {
      ...existing,
      ...details,
      completedAt: Date.now(),
      phase: 'completed'
    }
  })

  scheduleDismiss(key, COMPLETED_DISMISS_DELAY_MS)
}

export function setComputerUseError(sessionId: string | null | undefined, error: string): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  cancelDismiss(key)
  const sessions = $computerUseBySession.get()
  const existing = sessions[key]

  $computerUseBySession.set({
    ...sessions,
    [key]: {
      ...existing,
      completedAt: Date.now(),
      error,
      phase: 'error'
    }
  })

  scheduleDismiss(key, ERROR_DISMISS_DELAY_MS)
}

export function clearComputerUseState(
  sessionId: string | null | undefined,
  onlyIfUnfinished = false
): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  const sessions = $computerUseBySession.get()
  const existing = sessions[key]

  if (!existing) {
    return
  }

  // When onlyIfUnfinished is true (e.g. on message.complete), preserve completed or error
  // linger timer so user sees the dismissal animation and final outcome.
  if (onlyIfUnfinished && (existing.phase === 'completed' || existing.phase === 'error')) {
    return
  }

  cancelDismiss(key)

  const next = { ...sessions }
  delete next[key]
  $computerUseBySession.set(next)
}

export function clearAllComputerUseStates(): void {
  for (const timer of dismissTimers.values()) {
    clearTimeout(timer)
  }

  dismissTimers.clear()
  sessionComputerUseCache.clear()
  $computerUseBySession.set({})
}

/** Helper to extract structured computer_use args from gateway event payload without returning undefined fields */
export function extractComputerUseArgs(payload: unknown): Partial<ComputerUseActiveState> {
  if (!payload || typeof payload !== 'object') {
    return {}
  }

  const p = payload as Record<string, unknown>
  let rawArgs = p.args ?? p.arguments ?? p.parameters

  if (typeof rawArgs === 'string') {
    try {
      rawArgs = JSON.parse(rawArgs)
    } catch {
      rawArgs = {}
    }
  }

  const merged: Record<string, unknown> = {
    ...p,
    ...(rawArgs && typeof rawArgs === 'object' ? (rawArgs as Record<string, unknown>) : {})
  }

  const result: Partial<ComputerUseActiveState> = {}

  if (typeof merged.action === 'string' && merged.action.trim()) {
    result.action = merged.action.trim()
  }

  const rawApp =
    typeof merged.app === 'string'
      ? merged.app
      : typeof merged.window_title === 'string'
        ? merged.window_title
        : undefined

  if (rawApp && rawApp.trim()) {
    result.app = rawApp.trim()
  }

  if (typeof merged.mode === 'string' && merged.mode.trim()) {
    result.mode = merged.mode.trim()
  }

  if (typeof merged.element === 'number') {
    result.element = merged.element
  }

  if (typeof merged.text === 'string') {
    result.text = merged.text
  }

  if (typeof merged.keys === 'string') {
    result.keys = merged.keys
  }

  if (Array.isArray(merged.coordinate) && merged.coordinate.length >= 2) {
    const x = Number(merged.coordinate[0])
    const y = Number(merged.coordinate[1])

    if (!Number.isNaN(x) && !Number.isNaN(y)) {
      result.coordinate = [x, y]
    }
  }

  if (typeof merged.summary === 'string' && merged.summary.trim()) {
    result.targetSummary = merged.summary.trim()
  } else if (typeof merged.description === 'string' && merged.description.trim()) {
    result.targetSummary = merged.description.trim()
  } else if (typeof merged.progress === 'string' && merged.progress.trim()) {
    result.targetSummary = merged.progress.trim()
  }

  const rawToolId =
    typeof merged.tool_id === 'string'
      ? merged.tool_id
      : typeof merged.id === 'string'
        ? merged.id
        : undefined

  if (rawToolId && rawToolId.trim()) {
    result.toolId = rawToolId.trim()
  }

  return result
}

/**
 * Normalizes error messages from diverse gateway tool payloads, JSON-RPC 2.0 error responses,
 * CLI exit statuses, and standard Anthropic/MCP error structures.
 */
export function extractToolErrorMessage(payload: unknown): string | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const p = payload as Record<string, unknown>

  // 1. Direct error property
  if (p.error !== undefined && p.error !== null && p.error !== false) {
    if (typeof p.error === 'string' && p.error.trim()) {
      return cleanErrorMessage(p.error)
    }

    if (typeof p.error === 'object') {
      const errObj = p.error as Record<string, unknown>

      let msg: string | undefined

      if (typeof errObj.message === 'string' && errObj.message.trim()) {
        msg = errObj.message
      } else if (typeof errObj.error === 'string' && errObj.error.trim()) {
        msg = errObj.error
      } else if (typeof errObj.details === 'string' && errObj.details.trim()) {
        msg = errObj.details
      } else if (
        typeof errObj.name === 'string' &&
        errObj.name !== 'Error' &&
        errObj.name.trim()
      ) {
        msg = errObj.name
      } else {
        try {
          const str = JSON.stringify(errObj)
          msg = str && str !== '{}' ? str : 'Action failed'
        } catch {
          msg = 'Action failed'
        }
      }

      if (msg && msg.trim()) {
        return cleanErrorMessage(msg)
      }
    }

    if (p.error === true) {
      return 'Action failed'
    }

    return cleanErrorMessage(String(p.error))
  }

  // 2. Explicit boolean flags or status strings on payload
  const isErrorFlag =
    p.is_error === true ||
    p.isError === true ||
    p.status === 'error' ||
    p.status === 'failed' ||
    (typeof p.exit_code === 'number' && p.exit_code !== 0) ||
    (typeof p.exitCode === 'number' && p.exitCode !== 0)

  // 3. Inspect result payload (stringified JSON or object)
  if (p.result !== undefined && p.result !== null) {
    let resultObj: Record<string, unknown> | null = null

    if (typeof p.result === 'string') {
      const trimmed = p.result.trim()

      if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
        try {
          const parsed = JSON.parse(trimmed)

          if (parsed && typeof parsed === 'object') {
            resultObj = parsed as Record<string, unknown>
          }
        } catch {
          // not JSON
        }
      } else if (isErrorFlag) {
        return cleanErrorMessage(trimmed) || 'Action failed'
      } else if (/^(error|fatal|exception):/i.test(trimmed)) {
        return cleanErrorMessage(trimmed)
      }
    } else if (typeof p.result === 'object') {
      resultObj = p.result as Record<string, unknown>
    }

    if (resultObj) {
      if (resultObj.error !== undefined && resultObj.error !== null && resultObj.error !== false) {
        if (typeof resultObj.error === 'string' && resultObj.error.trim()) {
          return cleanErrorMessage(resultObj.error)
        }

        if (typeof resultObj.error === 'object') {
          const err = resultObj.error as Record<string, unknown>

          let msg: string | undefined

          if (typeof err.message === 'string' && err.message.trim()) {
            msg = err.message
          } else if (typeof err.error === 'string' && err.error.trim()) {
            msg = err.error
          } else if (typeof err.details === 'string' && err.details.trim()) {
            msg = err.details
          } else if (
            typeof err.name === 'string' &&
            err.name !== 'Error' &&
            err.name.trim()
          ) {
            msg = err.name
          } else {
            try {
              const str = JSON.stringify(err)
              msg = str && str !== '{}' ? str : 'Action failed'
            } catch {
              msg = 'Action failed'
            }
          }

          if (msg && msg.trim()) {
            return cleanErrorMessage(msg)
          }
        }

        return 'Action failed'
      }

      if (
        resultObj.ok === false ||
        resultObj.success === false ||
        resultObj.is_error === true ||
        resultObj.isError === true ||
        resultObj.status === 'error' ||
        resultObj.status === 'failed' ||
        (typeof resultObj.exit_code === 'number' && resultObj.exit_code !== 0) ||
        (typeof resultObj.exitCode === 'number' && resultObj.exitCode !== 0)
      ) {
        const msg =
          typeof resultObj.message === 'string' && resultObj.message.trim()
            ? resultObj.message
            : typeof resultObj.hint === 'string' && resultObj.hint.trim()
              ? resultObj.hint
              : typeof resultObj.details === 'string' && resultObj.details.trim()
                ? resultObj.details
                : 'Action failed'

        return cleanErrorMessage(msg)
      }
    }
  }

  if (isErrorFlag) {
    const fallbackMsg =
      typeof p.message === 'string' && p.message.trim()
        ? p.message
        : typeof p.hint === 'string' && p.hint.trim()
          ? p.hint
          : typeof p.details === 'string' && p.details.trim()
            ? p.details
            : 'Action failed'

    return cleanErrorMessage(fallbackMsg)
  }

  return undefined
}

export function cleanErrorMessage(raw: string): string {
  let cleaned = raw.trim()
  // Strip redundant leading "Error: " or "ToolError: " prefix
  cleaned = cleaned.replace(/^(error|toolerror|failederror):\s*/i, '')
  // Take first line if multi-line stack trace
  const firstNewline = cleaned.indexOf('\n')

  if (firstNewline > 0) {
    cleaned = cleaned.slice(0, firstNewline).trim()
  }

  // Truncate to reasonable length for pill display
  if (cleaned.length > 80) {
    cleaned = `${cleaned.slice(0, 77)}…`
  }

  return cleaned || 'Action failed'
}
