/**
 * Tell the backend when the user is at this window's session (`client.activity`), so a phone
 * notification plugin can stay quiet for a reply the user is here to see.
 *
 * A report is a claim that the user is at this computer AND this window can show them this
 * session's reply. A false claim silences a phone for a reply nobody sees, so whatever the reporter
 * last told a connection is corrected the moment it stops being true, and whenever it is not sure
 * (which backend, which session, whether it can alert) it claims nothing: the backend ages a record
 * by itself, so silence is always safe.
 *
 * Evidence, not focus: with the OS idle clock from main (`electron/presence.ts`) the evidence is
 * input anywhere on the computer; without it (the Webapp, Desktop on Linux) it is pointer/key/
 * wheel/touch input inside this document while it is visible. Focus and visibility gate reporting
 * but never count as input. Nothing here renders: the reporter lives in timers and listeners.
 */

import type { ClientActivityParams } from '@hermes/shared'

import type { PresenceSnapshot } from '../../electron/presence'

import { $browserNotificationPermission } from './browser-notification-permission'
import { $gateway } from './gateway'
import { $nativeNotifyPrefs, readHostNotifySupport, turnDoneNotificationWouldFire } from './native-notifications'
import { mainChatShowing, mainChatShowingStores, presenceView } from './presence-view'
import { $activeSessionId, $gatewayState, $selectedStoredSessionId } from './session'
import { $sessionStates } from './session-states'

/** Evidence younger than this counts as the user being here. */
export const PRESENCE_ACTIVE_SECONDS = 60
/** At most one report per this interval while nothing changes. */
export const PRESENCE_REPORT_MS = 45_000
/** How often the renderer re-reads its evidence; main's pushes also wake a throttled hidden window. */
const PRESENCE_TICK_MS = 15_000

/** Input here that ends an idle spell on the OS clock: let the OS count it, then ask main again. */
const PRESENCE_WAKE_MS = 250
/** At most one such question per this interval, for input the OS clock does not count. */
const PRESENCE_WAKE_GAP_MS = 1000

const PAGE_INPUT_EVENTS = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'] as const

export interface PresenceInputs {
  system: boolean
  /** OS idle seconds as of `now`; null without a system clock. */
  systemIdleSeconds: null | number
  locked: boolean
  suspended: boolean
  /** Last input inside this document, recorded only while it was visible (same clock as `now`). */
  lastPageInputAt: null | number
  visible: boolean
  focused: boolean
  now: number
}

export interface PresenceEvidence {
  active: boolean
  basis: ClientActivityParams['basis']
  /** Seconds since the newest input evidence; null when there is none. */
  idleSeconds: null | number
}

/** What this window can say about the user being here. */
export function presenceEvidence(inputs: PresenceInputs): PresenceEvidence {
  if (inputs.system) {
    const idleSeconds = inputs.systemIdleSeconds
    const active = !inputs.locked && !inputs.suspended && idleSeconds !== null && idleSeconds < PRESENCE_ACTIVE_SECONDS

    return { active, basis: 'system', idleSeconds }
  }

  const idleSeconds = inputs.lastPageInputAt === null ? null : Math.max(0, (inputs.now - inputs.lastPageInputAt) / 1000)
  // A hidden page cannot be where the user is looking, whatever its last input was.
  const active = inputs.visible && idleSeconds !== null && idleSeconds < PRESENCE_ACTIVE_SECONDS

  return { active, basis: 'page', idleSeconds }
}

export interface PresenceChannel {
  readonly clientActivity: boolean
  /** `'open'` only once the connection is negotiated; a report never waits for it. */
  readonly connectionState: string
  request<T>(method: string, params: Record<string, unknown>): Promise<T>
}

/** What this window presents, with the connection a submit for it would use (`presence-view.ts`). */
export type PresenceView =
  /** Nothing in view; `channel` is the window's own connection, for the readout only. */
  | { kind: 'none'; channel: PresenceChannel | null }
  /** In view, but the backend could not tell it from another session on the same socket. */
  | { kind: 'unroutable' }
  /** `channel` is null while the owning connection is not open. */
  | { kind: 'session'; channel: PresenceChannel | null; sessionId: string }

export interface PresenceSystemSource {
  get: () => Promise<PresenceSnapshot>
  onChange: (callback: (snapshot: PresenceSnapshot) => void) => () => void
}

export interface PresenceReporterDeps {
  view: () => PresenceView
  /** This client can get the user's attention for the session in view itself. */
  canAlert: (sessionId: string) => boolean
  visible: () => boolean
  focused: () => boolean
  /** Where page input is listened for (the document). */
  inputTarget: EventTarget
  system?: PresenceSystemSource
  /** Monotonic milliseconds: a wall-clock change must not make old input look fresh. */
  now?: () => number
}

/** The evidence the settings readout shows: its basis and how long ago the newest input was; null
 *  when none was seen yet. */
export interface PresenceInputAge {
  basis: PresenceEvidence['basis']
  idleSeconds: null | number
}

export interface PresenceReporter {
  /** Re-read the evidence (asking main for a fresh idle time when it has one) and report if due. */
  refresh: () => void
  evidence: () => PresenceEvidence
  inputAge: () => PresenceInputAge
  /** A connection reopened: whatever it was told is gone, so the next claim goes out at once. */
  reconnected: () => void
  stop: () => void
}

/** What the reporter last told one connection: a claim for `sessionId`, or nothing (null). */
interface Statement {
  at: number
  canAlert: boolean
  sessionId: null | string
}

const usable = (channel: null | PresenceChannel | undefined): channel is PresenceChannel =>
  channel?.connectionState === 'open' && channel.clientActivity === true

export function createPresenceReporter(deps: PresenceReporterDeps): PresenceReporter {
  const now = deps.now ?? (() => performance.now())
  const startedAt = now()
  let snapshot: PresenceSnapshot = {
    system: false,
    idleSeconds: null,
    locked: false,
    suspended: false,
    seq: -1,
    sampledAt: 0
  }
  // Idle seconds as of `snapshotAt`, after the delivery delay.
  let snapshotIdle: null | number = null
  let snapshotAt = startedAt
  let lastPageInputAt: null | number = null
  let active = false
  let pendingEvaluate: ReturnType<typeof setTimeout> | null = null
  let wakeAskedAt = Number.NEGATIVE_INFINITY
  const told = new Map<PresenceChannel, Statement>()
  // The newest statement each connection accepted, with its send order: what it holds after a failure.
  const confirmed = new Map<PresenceChannel, { order: number; statement: Statement }>()
  let sent = 0
  // Connections already told `away` in this lock/suspend episode.
  const awayTold = new Set<PresenceChannel>()

  const evidence = (): PresenceEvidence => {
    const at = now()

    return presenceEvidence({
      ...snapshot,
      systemIdleSeconds: snapshotIdle === null ? null : snapshotIdle + (at - snapshotAt) / 1000,
      lastPageInputAt,
      visible: deps.visible(),
      focused: deps.focused(),
      now: at
    })
  }

  const forget = (channel: PresenceChannel) => {
    told.delete(channel)
    confirmed.delete(channel)
  }

  // Checked at the moment of sending, never queued: a connection that is not ready drops the report,
  // and the next evaluation after it is ready sends a fresh one.
  const send = (channel: PresenceChannel, params: ClientActivityParams, statement: Statement) => {
    if (!usable(channel)) {
      forget(channel)

      return
    }

    const order = (sent += 1)
    told.set(channel, statement)
    // A refused or failed report must never reach the user or the connection. It was not told,
    // though: the connection holds the newest statement it accepted (an older one still in flight
    // or failed is not that), and the next evaluation corrects it again. Only while the failed one
    // is still the newest statement, so a late failure cannot undo a later report.
    channel.request('client.activity', { ...params }).then(
      () => {
        if ((confirmed.get(channel)?.order ?? 0) < order) {
          confirmed.set(channel, { order, statement })
        }
      },
      () => {
        if (told.get(channel) !== statement) {
          return
        }

        const held = confirmed.get(channel)?.statement

        if (held) {
          told.set(channel, held)
        } else {
          told.delete(channel)
        }

        awayTold.delete(channel)
      }
    )
  }

  // An empty list withdraws everything this connection holds without claiming any input.
  const withdraw = (channel: PresenceChannel, current: PresenceEvidence, away = false) =>
    send(
      channel,
      {
        ...(away && { away: true }),
        basis: current.basis,
        idle_seconds: current.idleSeconds ?? PRESENCE_ACTIVE_SECONDS,
        sessions: []
      },
      { at: now(), canAlert: false, sessionId: null }
    )

  // Away once per episode on every connection that holds a statement or owns the session in view
  // (server-seen input, a submit, may be there even before any report of ours).
  const reportAway = (view: PresenceView, current: PresenceEvidence) => {
    const targets = new Set(told.keys())

    if (view.kind !== 'unroutable' && view.channel) {
      targets.add(view.channel)
    }

    for (const channel of targets) {
      if (usable(channel) && !awayTold.has(channel)) {
        awayTold.add(channel)
        withdraw(channel, current, true)
      }
    }
  }

  const claim = (channel: PresenceChannel, sessionId: string, current: PresenceEvidence, wasActive: boolean) => {
    const last = told.get(channel)
    const canAlert = deps.canAlert(sessionId)

    if (current.active && current.idleSeconds !== null) {
      const due =
        !wasActive ||
        last?.sessionId !== sessionId ||
        last.canAlert !== canAlert ||
        now() - last.at >= PRESENCE_REPORT_MS

      if (due) {
        send(
          channel,
          { basis: current.basis, can_alert: canAlert, idle_seconds: current.idleSeconds, sessions: [sessionId] },
          { at: now(), canAlert, sessionId }
        )
      }

      return
    }

    // Stale evidence: nothing new to claim, but a claim that stopped being true is withdrawn.
    if (last?.sessionId && (last.sessionId !== sessionId || (last.canAlert && !canAlert))) {
      withdraw(channel, current)
    }
  }

  const evaluate = () => {
    const current = evidence()
    const wasActive = active
    const view = deps.view()
    const away = snapshot.locked || snapshot.suspended
    active = current.active

    for (const channel of told.keys()) {
      if (!usable(channel)) {
        forget(channel)
      }
    }

    if (away) {
      return reportAway(view, current)
    }

    awayTold.clear()
    const target = view.kind === 'session' && usable(view.channel) ? view.channel : null

    for (const [channel, statement] of told) {
      if (channel !== target && statement.sessionId !== null) {
        withdraw(channel, current)
      }
    }

    if (target && view.kind === 'session') {
      claim(target, view.sessionId, current, wasActive)
    }
  }

  // Main samples in order; a reply or push older than the newest adopted one is ignored, so a late
  // pre-lock answer can never undo a lock.
  const adopt = (next: PresenceSnapshot) => {
    if (next.seq < snapshot.seq) {
      return
    }

    // Page input from before a sleep is no evidence after it, however the clock counted the sleep.
    if (snapshot.suspended && !next.suspended) {
      lastPageInputAt = null
    }

    snapshot = next
    // A late delivery to a throttled window must not read as fresher than it is.
    snapshotIdle = next.idleSeconds === null ? null : next.idleSeconds + Math.max(0, Date.now() - next.sampledAt) / 1000
    snapshotAt = now()
    evaluate()
  }

  let stopped = false

  // A held snapshot ages between main's pushes; asking again keeps idle time exact.
  const refresh = () => {
    if (!deps.system) {
      return evaluate()
    }

    deps.system.get().then(
      next => !stopped && adopt(next),
      () => !stopped && evaluate()
    )
  }

  // Passive, capture-phase, and only a timestamp: the report itself is rate-limited above. The one
  // exception is input that ends an idle spell, which must report now rather than at the next tick.
  // With an OS clock that clock stays the only evidence: main is asked again once the OS has had a
  // moment to count this input, instead of waiting for main's next poll.
  const onPageInput = () => {
    if (!deps.visible()) {
      return
    }

    lastPageInputAt = now()

    if (active || pendingEvaluate !== null) {
      return
    }

    const wait = snapshot.system ? Math.max(PRESENCE_WAKE_MS, wakeAskedAt + PRESENCE_WAKE_GAP_MS - now()) : 0

    pendingEvaluate = setTimeout(() => {
      pendingEvaluate = null

      if (!snapshot.system) {
        return evaluate()
      }

      wakeAskedAt = now()
      refresh()
    }, wait)
  }

  const listenerOptions: AddEventListenerOptions = { capture: true, passive: true }

  for (const type of PAGE_INPUT_EVENTS) {
    deps.inputTarget.addEventListener(type, onPageInput, listenerOptions)
  }

  const inputAge = (): PresenceInputAge => {
    const { basis, idleSeconds } = evidence()

    return { basis, idleSeconds }
  }

  const offSystem = deps.system?.onChange(adopt)
  refresh()
  const ticker = setInterval(refresh, PRESENCE_TICK_MS)

  return {
    refresh,
    evidence,
    inputAge,
    reconnected: () => {
      for (const statement of told.values()) {
        statement.at = Number.NEGATIVE_INFINITY
      }

      // A reopened connection holds nothing it accepted before.
      confirmed.clear()
    },
    stop: () => {
      stopped = true
      clearInterval(ticker)

      if (pendingEvaluate !== null) {
        clearTimeout(pendingEvaluate)
      }

      offSystem?.()

      for (const type of PAGE_INPUT_EVENTS) {
        deps.inputTarget.removeEventListener(type, onPageInput, listenerOptions)
      }
    }
  }
}

const documentVisible = () => document.visibilityState === 'visible'
const documentFocused = () => document.hasFocus()

let windowReporter: PresenceReporter | null = null

/** This window's reporter, for the settings readout; null until the gateway boot starts it. */
export function windowPresenceReporter(): PresenceReporter | null {
  return windowReporter
}

/** Re-read what is in view now: for a view change no store or DOM event announces (a router move). */
export function refreshWindowPresence(): void {
  windowReporter?.refresh()
}

/** Start this window's reporter; called once by the hook that owns the gateway connection. */
export function startPresenceReporter(): () => void {
  const reporter = createPresenceReporter({
    // The user sees the reply on screen, or this client would hand its own notification to a host
    // that can show it.
    canAlert: sessionId =>
      (documentVisible() && documentFocused() && mainChatShowing()) || turnDoneNotificationWouldFire(sessionId),
    focused: documentFocused,
    inputTarget: document,
    system: window.hermesDesktop?.presence,
    view: presenceView,
    visible: documentVisible
  })

  const onView = () => reporter.refresh()
  let stopped = false
  // Main's answer about OS notifications decides can_alert in a background window: ask now, not at
  // the first blur, and read presence again once it is in.
  void readHostNotifySupport()?.then(() => !stopped && onView())
  const boundStored = () => $sessionStates.get()[$activeSessionId.get() ?? '']?.storedSessionId
  let boundStoredId = boundStored()
  let chatShowing = mainChatShowing()

  const offs = [
    $activeSessionId.listen(onView),
    $selectedStoredSessionId.listen(onView),
    $gateway.listen(onView),
    // Only the active runtime's binding matters; the map itself changes with every stream delta.
    $sessionStates.listen(() => {
      const next = boundStored()

      if (next !== boundStoredId) {
        boundStoredId = next
        onView()
      }
    }),
    // The layout changes with every drag; only the main chat going behind or in front matters.
    ...mainChatShowingStores().map(store =>
      store.listen(() => {
        const next = mainChatShowing()

        if (next !== chatShowing) {
          chatShowing = next
          onView()
        }
      })
    ),
    $nativeNotifyPrefs.listen(onView),
    $browserNotificationPermission.listen(onView),
    $gatewayState.listen(state => {
      // Every (re)connection starts with an empty presence record on the backend.
      if (state === 'open') {
        reporter.reconnected()
      }

      onView()
    })
  ]

  // The route on screen decides what is in view. In-app moves arrive through usePresenceRoute (the
  // router pushes history without a hashchange); this covers a hash edited from outside the router.
  window.addEventListener('hashchange', onView)
  window.addEventListener('focus', onView)
  window.addEventListener('blur', onView)
  document.addEventListener('visibilitychange', onView)
  windowReporter = reporter

  return () => {
    stopped = true
    offs.forEach(off => off())
    window.removeEventListener('hashchange', onView)
    window.removeEventListener('focus', onView)
    window.removeEventListener('blur', onView)
    document.removeEventListener('visibilitychange', onView)
    reporter.stop()

    if (windowReporter === reporter) {
      windowReporter = null
    }
  }
}
