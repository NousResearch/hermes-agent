import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  dispatchNativeNotification,
  NATIVE_NOTIFICATION_KINDS,
  setNativeNotifyEnabled,
  setNativeNotifyKind
} from './native-notifications'
import { setActiveSessionId } from './session'
import { $splitPaneRuntimeSessionId, registerMainPaneRuntimeSessionIdGetter } from './split'

// Step 14's foreground amendment: a session visible in the SPLIT pane is as
// on-screen as the active one — completion kinds may fire for it while the
// user is away, and attention kinds stop breaking through while the user is
// literally looking at it. With the split closed the runtime-id atom is null
// and every path below reduces to the pinned single-pane suite.

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

const notify = vi.fn().mockResolvedValue(true)

function setWindowState({ focused = true, hidden = false }: { focused?: boolean; hidden?: boolean }) {
  Object.defineProperty(document, 'hidden', { configurable: true, value: hidden })
  Object.defineProperty(document, 'hasFocus', { configurable: true, value: () => focused })
}

let counter = 0

// Unique session id per call dodges the per-(kind,session) throttle so each
// assertion starts clean.
function freshSession(): string {
  counter += 1

  return `split-session-${counter}`
}

beforeEach(() => {
  notify.mockClear()
  desktopWindow.hermesDesktop = { notify } as unknown as Window['hermesDesktop']
  setNativeNotifyEnabled(true)

  for (const kind of NATIVE_NOTIFICATION_KINDS) {
    setNativeNotifyKind(kind, true)
  }

  setActiveSessionId('rt-main')
  $splitPaneRuntimeSessionId.set(null)
  setWindowState({ focused: false, hidden: true })
})

afterEach(() => {
  setActiveSessionId(null)
  $splitPaneRuntimeSessionId.set(null)

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('split-visible sessions count as foreground', () => {
  it('fires a completion notification for the split session while the user is away', () => {
    const sessionId = freshSession()
    $splitPaneRuntimeSessionId.set(sessionId)

    dispatchNativeNotification({ kind: 'turnDone', sessionId, title: 'done' })

    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('still suppresses completion notifications for genuinely background sessions', () => {
    $splitPaneRuntimeSessionId.set(freshSession())

    dispatchNativeNotification({ kind: 'turnDone', sessionId: freshSession(), title: 'done' })

    expect(notify).not.toHaveBeenCalled()
  })

  it('does NOT break an attention notification through for the split session while focused', () => {
    const sessionId = freshSession()
    $splitPaneRuntimeSessionId.set(sessionId)
    setWindowState({ focused: true, hidden: false })

    // Pre-amendment this fired: the session wasn't the ACTIVE one, so the
    // attention kind treated it as off-screen even though it fills a pane.
    dispatchNativeNotification({ kind: 'approval', sessionId, title: 'approve?' })

    expect(notify).not.toHaveBeenCalled()
  })

  it('keeps breaking attention kinds through for off-screen sessions while focused', () => {
    $splitPaneRuntimeSessionId.set(freshSession())
    setWindowState({ focused: true, hidden: false })

    dispatchNativeNotification({ kind: 'approval', sessionId: freshSession(), title: 'approve?' })

    expect(notify).toHaveBeenCalledTimes(1)
  })

  it('split closed (null runtime id): completion gating is exactly the single-pane rule', () => {
    dispatchNativeNotification({ kind: 'turnDone', sessionId: 'rt-main', title: 'done' })
    expect(notify).toHaveBeenCalledTimes(1)

    notify.mockClear()
    dispatchNativeNotification({ kind: 'turnDone', sessionId: freshSession(), title: 'done' })
    expect(notify).not.toHaveBeenCalled()
  })
})

describe('main pane stays foreground while the SPLIT is focused (identity mirrored)', () => {
  // While the split is focused, split-mirror points $activeSessionId at the
  // SPLIT's runtime id; the main pane's on-screen session is then only
  // reachable through the controller-registered unmirrored getter.
  afterEach(() => registerMainPaneRuntimeSessionIdGetter(null))

  it('does NOT break an attention notification through for the main pane session', () => {
    const mainSessionId = freshSession()
    const splitSessionId = freshSession()
    // Mirror in effect: the ACTIVE singleton carries the split's id.
    setActiveSessionId(splitSessionId)
    $splitPaneRuntimeSessionId.set(splitSessionId)
    registerMainPaneRuntimeSessionIdGetter(() => mainSessionId)
    setWindowState({ focused: true, hidden: false })

    dispatchNativeNotification({ kind: 'approval', sessionId: mainSessionId, title: 'approve?' })

    expect(notify).not.toHaveBeenCalled()
  })

  it('fires a completion notification for the main pane session while the user is away', () => {
    const mainSessionId = freshSession()
    const splitSessionId = freshSession()
    setActiveSessionId(splitSessionId)
    $splitPaneRuntimeSessionId.set(splitSessionId)
    registerMainPaneRuntimeSessionIdGetter(() => mainSessionId)
    setWindowState({ focused: false, hidden: true })

    dispatchNativeNotification({ kind: 'turnDone', sessionId: mainSessionId, title: 'done' })

    expect(notify).toHaveBeenCalledTimes(1)
  })
})
