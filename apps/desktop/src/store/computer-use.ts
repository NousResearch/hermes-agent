import { atom, computed, type ReadableAtom } from 'nanostores'

import type { Translations } from '@/i18n'

import { $activeSessionId } from './session'

export type ComputerUsePhase = 'drafting' | 'running' | 'completed' | 'error'

export type ComputerUseLabels = Translations['titlebar']['computerUse']

/** What the pill shows about one computer_use call, read from its arguments. */
export interface ComputerUseCall {
  toolId?: string
  action?: string
  app?: string
  mode?: string
  element?: number
  coordinate?: [number, number]
  /** Only the length: the text the agent types (passwords, codes) never reaches the titlebar. */
  textLength?: number
  keys?: string
}

export interface ComputerUseActiveState extends ComputerUseCall {
  phase: ComputerUsePhase
  startedAt?: number
  completedAt?: number
  durationSeconds?: number
  /** Cleaned failure detail; absent when the tool failed without saying why. */
  error?: string
}

const keyFor = (sessionId: string | null | undefined): string => sessionId ?? ''

const isSettled = (state: ComputerUseActiveState) => state.phase === 'completed' || state.phase === 'error'

export const $computerUseBySession = atom<Record<string, ComputerUseActiveState>>({})

/**
 * Normalizes application / window identifiers.
 * E.g. "C:\\Windows\\notepad.exe" -> "Notepad", "chrome.exe" -> "Chrome".
 */
export function cleanAppLabel(rawApp: string | undefined): string {
  if (!rawApp) {
    return ''
  }

  let name = rawApp.trim()

  // Extract basename if full filesystem path was passed
  const lastSlash = Math.max(name.lastIndexOf('/'), name.lastIndexOf('\\'))

  if (lastSlash >= 0) {
    name = name.slice(lastSlash + 1)
  }

  // Strip .exe or .app extensions
  name = name.replace(/\.(exe|app)$/i, '')

  // Capitalize first character if single-word lowercase
  if (name && name[0] === name[0].toLowerCase() && !name.includes(' ')) {
    name = name.charAt(0).toUpperCase() + name.slice(1)
  }

  return name
}

function appDisplayName(app: string | undefined, labels: ComputerUseLabels): string {
  const sentinel = app?.trim().toLowerCase()

  // app='screen' / app='desktop' are capture sentinels, not app names.
  if (sentinel === 'screen' || sentinel === 'desktop') {
    return labels[sentinel]
  }

  return cleanAppLabel(app)
}

function actionDetail(call: ComputerUseCall, labels: ComputerUseLabels): string {
  switch (call.action) {
    case 'capture':
      return call.mode ? `(${call.mode.toUpperCase()})` : ''

    case 'type':
      return call.textLength != null ? `(${labels.typedChars(call.textLength)})` : ''

    case 'key':
      return call.keys ? `(${call.keys})` : ''

    default:
      if (call.element != null) {
        return `#${call.element}`
      }

      return call.coordinate ? `(${call.coordinate[0]}, ${call.coordinate[1]})` : ''
  }
}

/**
 * Concise, localized summary of a computer_use call.
 * E.g. "Chrome · Click #12", "Notepad · Type (5 characters)", "Screen · Capture (SOM)".
 */
export function formatComputerUseTarget(call: ComputerUseCall, labels: ComputerUseLabels): string {
  const { action } = call
  const appLabel = appDisplayName(call.app, labels)

  const actionName = action
    ? Object.hasOwn(labels.actions, action)
      ? labels.actions[action as keyof ComputerUseLabels['actions']]
      : action.replace(/_/g, ' ')
    : ''

  const detail = action ? actionDetail(call, labels) : ''
  const actionLabel = [actionName, detail].filter(Boolean).join(' ')

  return [appLabel, actionLabel].filter(Boolean).join(' · ') || labels.active
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

function scheduleDismiss(sessionId: string, delayMs: number): void {
  cancelDismiss(sessionId)

  const timer = setTimeout(() => {
    dismissTimers.delete(sessionId)
    clearComputerUseState(sessionId)
  }, delayMs)

  dismissTimers.set(sessionId, timer)
}

function writeState(key: string, state: ComputerUseActiveState): void {
  $computerUseBySession.set({ ...$computerUseBySession.get(), [key]: state })
}

export function setComputerUseDrafting(sessionId: string | null | undefined): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  cancelDismiss(key)
  writeState(key, { phase: 'drafting', startedAt: Date.now() })
}

/**
 * A computer_use call started. It replaces whatever the previous call left
 * behind — its element, coordinates, error and dismiss timer describe a
 * different action and must not leak into this one's label.
 */
export function setComputerUseRunning(sessionId: string | null | undefined, call: ComputerUseCall): void {
  const key = keyFor(sessionId)

  if (!key) {
    return
  }

  cancelDismiss(key)
  writeState(key, { ...call, phase: 'running', startedAt: Date.now() })
}

/** Settles only a call that is still live: a retired or already-settled one stays as it is. */
function settleableState(key: string): ComputerUseActiveState | undefined {
  const existing = key ? $computerUseBySession.get()[key] : undefined

  return existing && !isSettled(existing) ? existing : undefined
}

export function setComputerUseCompleted(
  sessionId: string | null | undefined,
  details: { durationSeconds?: number } = {}
): void {
  const key = keyFor(sessionId)
  const existing = settleableState(key)

  if (!existing) {
    return
  }

  writeState(key, { ...existing, ...details, completedAt: Date.now(), phase: 'completed' })
  scheduleDismiss(key, COMPLETED_DISMISS_DELAY_MS)
}

export function setComputerUseError(sessionId: string | null | undefined, error?: string): void {
  const key = keyFor(sessionId)
  const existing = settleableState(key)

  if (!existing) {
    return
  }

  writeState(key, { ...existing, completedAt: Date.now(), error, phase: 'error' })
  scheduleDismiss(key, ERROR_DISMISS_DELAY_MS)
}

export function clearComputerUseState(sessionId: string | null | undefined, onlyIfUnfinished = false): void {
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
  if (onlyIfUnfinished && isSettled(existing)) {
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

function parseArgs(rawArgs: unknown): Record<string, unknown> {
  if (typeof rawArgs === 'string') {
    try {
      rawArgs = JSON.parse(rawArgs)
    } catch {
      return {}
    }
  }

  return rawArgs && typeof rawArgs === 'object' ? (rawArgs as Record<string, unknown>) : {}
}

const nonEmpty = (value: unknown): string | undefined =>
  typeof value === 'string' && value.trim() ? value.trim() : undefined

/** The pill-relevant fields of a computer_use tool event, without undefined entries. */
export function extractComputerUseArgs(payload: unknown): ComputerUseCall {
  if (!payload || typeof payload !== 'object') {
    return {}
  }

  const p = payload as Record<string, unknown>
  const args = parseArgs(p.args ?? p.arguments ?? p.parameters)
  const result: ComputerUseCall = {}

  const toolId = nonEmpty(p.tool_id) ?? nonEmpty(p.id)
  const action = nonEmpty(args.action)
  const app = nonEmpty(args.app) ?? nonEmpty(args.window_title)
  const mode = nonEmpty(args.mode)

  if (toolId) {
    result.toolId = toolId
  }

  if (action) {
    result.action = action
  }

  if (app) {
    result.app = app
  }

  if (mode) {
    result.mode = mode
  }

  if (typeof args.element === 'number') {
    result.element = args.element
  }

  if (typeof args.text === 'string') {
    result.textLength = [...args.text].length
  }

  if (typeof args.keys === 'string') {
    result.keys = args.keys
  }

  if (Array.isArray(args.coordinate) && args.coordinate.length >= 2) {
    const x = Number(args.coordinate[0])
    const y = Number(args.coordinate[1])

    if (!Number.isNaN(x) && !Number.isNaN(y)) {
      result.coordinate = [x, y]
    }
  }

  return result
}

const hasFailureFlag = (obj: Record<string, unknown>): boolean =>
  obj.is_error === true ||
  obj.isError === true ||
  obj.status === 'error' ||
  obj.status === 'failed' ||
  (typeof obj.exit_code === 'number' && obj.exit_code !== 0) ||
  (typeof obj.exitCode === 'number' && obj.exitCode !== 0)

const firstText = (obj: Record<string, unknown>, keys: readonly string[]): string | undefined =>
  keys.map(key => obj[key]).find((value): value is string => typeof value === 'string' && Boolean(value.trim()))

/** A failed tool call; `message` is absent when the payload gives no usable reason. */
export interface ToolFailure {
  message?: string
}

function failure(raw: string | undefined): ToolFailure {
  const message = raw === undefined ? '' : cleanErrorMessage(raw)

  return message ? { message } : {}
}

function errorObjectMessage(errObj: Record<string, unknown>): string | undefined {
  const text = firstText(errObj, ['message', 'error', 'details'])

  if (text) {
    return text
  }

  if (typeof errObj.name === 'string' && errObj.name !== 'Error' && errObj.name.trim()) {
    return errObj.name
  }

  try {
    const str = JSON.stringify(errObj)

    return str && str !== '{}' ? str : undefined
  } catch {
    return undefined
  }
}

/**
 * Detects a failed tool call across gateway payload shapes, JSON-RPC 2.0
 * errors, CLI exit statuses, and MCP error structures. Undefined = success.
 */
export function extractToolFailure(payload: unknown): ToolFailure | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const p = payload as Record<string, unknown>

  // 1. Direct error property
  if (p.error !== undefined && p.error !== null && p.error !== false) {
    if (typeof p.error === 'object') {
      return failure(errorObjectMessage(p.error as Record<string, unknown>))
    }

    return p.error === true ? {} : failure(String(p.error))
  }

  // 2. Explicit boolean flags or status strings on payload
  const isErrorFlag = hasFailureFlag(p)

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
      } else if (isErrorFlag || /^(error|fatal|exception):/i.test(trimmed)) {
        return failure(trimmed)
      }
    } else if (typeof p.result === 'object') {
      resultObj = p.result as Record<string, unknown>
    }

    if (resultObj) {
      if (resultObj.error !== undefined && resultObj.error !== null && resultObj.error !== false) {
        if (typeof resultObj.error === 'string') {
          return failure(resultObj.error)
        }

        return typeof resultObj.error === 'object'
          ? failure(errorObjectMessage(resultObj.error as Record<string, unknown>))
          : {}
      }

      if (resultObj.ok === false || resultObj.success === false || hasFailureFlag(resultObj)) {
        return failure(firstText(resultObj, ['message', 'hint', 'details']))
      }
    }
  }

  return isErrorFlag ? failure(firstText(p, ['message', 'hint', 'details'])) : undefined
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

  return cleaned
}
