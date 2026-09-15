/** Cursor Ink conversation scroll is `U()`, not xterm history.
 *  Wheel over a TUI must be stolen from xterm (or it shows old PTY frames)
 *  and forwarded as Page Up/Down — keys Ink already maps. Digit CSI (9001)
 *  leaks into the prompt. C-u/arrows/SGR are not used. */

export const TERMINAL_WHEEL_FIT_FREEZE_MS = 500
export const TERMINAL_WHEEL_SCROLLBACK = 1000

export function shouldFreezeTerminalFitForWheel(
  now: number,
  lastWheelAt: number,
  freezeMs = TERMINAL_WHEEL_FIT_FREEZE_MS
): boolean {
  return lastWheelAt > 0 && now - lastWheelAt < freezeMs
}

export function isShellTitleForWheel(title: string): boolean {
  const value = title.trim()

  if (!value) {
    return false
  }

  if (/^(zsh|bash|sh|fish|pwsh|powershell|cmd|login)(\b|$)/i.test(value)) {
    return true
  }

  if (/^[\w.-]+@[\w.-]+:/.test(value)) {
    return true
  }

  return /^~(\/|$)/.test(value) || /^\/[A-Za-z0-9._-]+/.test(value)
}

export function isTuiTitleForWheel(title: string | undefined): boolean {
  if (!title?.trim()) {
    return false
  }

  return !isShellTitleForWheel(title)
}

export function isCursorChromeForWheel(chunk: string): boolean {
  return /Run Everything|ctrl\+c to stop|Cursor Grok|cursor-agent|Cursor Agent/i.test(chunk)
}

export function nextTuiWheelLatch(
  latched: boolean,
  bufferType: string | undefined,
  title = '',
  chrome = '',
  allowUnlatch = true
): boolean {
  if (bufferType === 'alternate' || isTuiTitleForWheel(title) || isCursorChromeForWheel(chrome)) {
    return true
  }

  if (allowUnlatch && isShellTitleForWheel(title) && bufferType !== 'alternate') {
    return false
  }

  return latched
}

export function shouldSendTuiWheelToPty(
  bufferType: string | undefined,
  title = '',
  shiftKey = false,
  latched = false,
  allowUnlatch = true
): boolean {
  if (shiftKey) {
    return false
  }

  return nextTuiWheelLatch(latched, bufferType, title, '', allowUnlatch)
}

export function isPointInTerminalWheelRect(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, 'bottom' | 'height' | 'left' | 'right' | 'top' | 'width'>
): boolean {
  if (rect.width <= 0 || rect.height <= 0) {
    return false
  }

  return clientX >= rect.left && clientX <= rect.right && clientY >= rect.top && clientY <= rect.bottom
}

export function isTerminalWheelEventTarget(target: EventTarget | null): boolean {
  return target instanceof Element && Boolean(target.closest('[data-terminal], [data-persistent-terminal]'))
}

export interface TuiWheelPlanInput {
  blocked?: boolean
  bufferType?: string
  deltaMode?: number
  deltaY: number
  latched?: boolean
  pixelCarry?: number
  sessionId?: null | string
  shiftKey?: boolean
  sticky?: boolean
  title?: string
}

export interface TuiWheelPlan {
  handled: boolean
  nextCarry: number
  nextLatched: boolean
  sequence: string
}

/** One Desktop wheel tick: latch, steal, Page Up/Down only. */
export function planTuiWheelWrite({
  blocked = false,
  bufferType,
  deltaMode = 0,
  deltaY,
  latched = false,
  pixelCarry = 0,
  sessionId,
  shiftKey = false,
  sticky = false,
  title = ''
}: TuiWheelPlanInput): TuiWheelPlan {
  const nextLatched = sticky || nextTuiWheelLatch(latched, bufferType, title, '', false)

  if (!sessionId || shiftKey || blocked) {
    return { handled: false, nextCarry: pixelCarry, nextLatched, sequence: '' }
  }

  if (!sticky && !shouldSendTuiWheelToPty(bufferType, title, shiftKey, nextLatched, false)) {
    return { handled: false, nextCarry: 0, nextLatched, sequence: '' }
  }

  if (deltaMode === 0) {
    const carry = pixelCarry && deltaY && Math.sign(pixelCarry) !== Math.sign(deltaY) ? 0 : pixelCarry
    const consumed = consumePixelWheelCarry(carry + deltaY)

    return { handled: true, nextCarry: consumed.remaining, nextLatched, sequence: consumed.sequence }
  }

  return {
    handled: true,
    nextCarry: 0,
    nextLatched,
    sequence: encodeInkWheelToPty(deltaY, deltaMode)
  }
}

export interface TuiWheelDispatcherOptions {
  getBufferType: () => string | undefined
  getHost: () => HTMLElement
  getLatched: () => boolean
  getPixelCarry: () => number
  getSessionId: () => null | string
  getTitle: () => string
  isSticky?: () => boolean
  scrollToBottom: () => void
  setLatched: (value: boolean) => void
  setPixelCarry: (value: number) => void
  write: (id: string, data: string) => void
}

export function createTuiWheelDispatcher(options: TuiWheelDispatcherOptions) {
  let handledWheel: WheelEvent | null = null

  const send = (event: WheelEvent): boolean => {
    if (handledWheel === event) {
      return event.defaultPrevented
    }

    const host = options.getHost()
    const overHost =
      isPointInTerminalWheelRect(event.clientX, event.clientY, host.getBoundingClientRect()) ||
      isTerminalWheelEventTarget(event.target)

    if (!overHost) {
      return false
    }

    const target = event.target
    const plan = planTuiWheelWrite({
      blocked: target instanceof Element && Boolean(target.closest('button, [role="dialog"], [data-no-tui-wheel]')),
      bufferType: options.getBufferType(),
      deltaMode: event.deltaMode,
      deltaY: event.deltaY,
      latched: options.getLatched(),
      pixelCarry: options.getPixelCarry(),
      sessionId: options.getSessionId(),
      shiftKey: event.shiftKey,
      sticky: options.isSticky?.() ?? false,
      title: options.getTitle()
    })
    options.setLatched(plan.nextLatched)
    options.setPixelCarry(plan.nextCarry)

    if (!plan.handled) {
      return false
    }

    handledWheel = event
    event.preventDefault()
    event.stopPropagation()
    options.scrollToBottom()

    const id = options.getSessionId()

    if (id && plan.sequence) {
      options.write(id, plan.sequence)
    }

    return true
  }

  const onWindowWheel = (event: WheelEvent) => {
    if (!options.getHost().isConnected) {
      return
    }

    send(event)
  }

  return { onWindowWheel, send }
}

/** One conversation page per this many pixel-mode wheel units.
 *  Trackpads emit many tiny deltas; keep this low enough that a short flick
 *  still reaches Ink instead of feeling like a dead scroll area. */
export const TERMINAL_WHEEL_PIXEL_PAGE = 8
export const INK_PAGE_UP = '\x1b[5~'
export const INK_PAGE_DOWN = '\x1b[6~'

export function inkPageKey(direction: -1 | 1): string {
  return direction < 0 ? INK_PAGE_UP : INK_PAGE_DOWN
}

/** Page Up / Page Down only. Ink treats unknown `9001` digits as a count prefix.
 *  Pixel-mode trackpads emit many tiny deltas; those must accumulate to a page
 *  or every 2px tick becomes a full `U()` jump and the transcript looks stuck. */
export function encodeInkWheelToPty(deltaY: number, deltaMode = 0): string {
  if (!deltaY || !Number.isFinite(deltaY)) {
    return ''
  }

  const key = deltaY < 0 ? '\x1b[5~' : '\x1b[6~'
  let steps = 0

  if (deltaMode === 1) {
    steps = Math.round(Math.abs(deltaY))
  } else if (deltaMode === 2) {
    steps = 2
  } else {
    steps = Math.floor(Math.abs(deltaY) / TERMINAL_WHEEL_PIXEL_PAGE)
  }

  if (steps < 1) {
    return ''
  }

  return key.repeat(Math.min(2, steps))
}

/** Apply a pixel-mode tick to a running carry and emit at most two page keys. */
export function consumePixelWheelCarry(carry: number): { remaining: number; sequence: string } {
  const sequence = encodeInkWheelToPty(carry, 0)

  if (!sequence) {
    return { remaining: carry, sequence: '' }
  }

  const consumed = Math.trunc(carry / TERMINAL_WHEEL_PIXEL_PAGE) * TERMINAL_WHEEL_PIXEL_PAGE

  return { remaining: carry - consumed, sequence }
}

export function tuiXtermScrollback(isTui: boolean): number {
  return isTui ? 0 : TERMINAL_WHEEL_SCROLLBACK
}
