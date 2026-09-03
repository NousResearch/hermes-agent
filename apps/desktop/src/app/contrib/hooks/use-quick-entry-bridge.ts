import { useEffect, useRef } from 'react'

import { agentEmoji } from '@/lib/agent-emoji'
import type { HudRoomFeed } from '@/lib/hud-prefs'
import { analyzePromptDraft } from '@/lib/prompt-coach'
import { enhancePromptCoachWithAI } from '@/lib/prompt-coach-ai'
import { openHudForProfile } from '@/store/hud'
import {
  $profileColors,
  $profileOrder,
  $profiles,
  normalizeProfileKey,
  profileLabel,
  sortByProfileOrder
} from '@/store/profile'
import {
  initQuickEntryBridge,
  QUICK_ENTRY_FALLBACK_AGENTS,
  QUICK_TARGET_CURRENT,
  QUICK_TARGET_NEW,
  quickEntryAgentDisplayName,
  type QuickEntryAgentOption,
  type QuickEntryGroupOption,
  type QuickEntrySessionOption,
  setQuickEntrySubmitHandler
} from '@/store/quick-entry'
import { $gatewayState, $sessions } from '@/store/session'
import { sessionTileDelegate } from '@/store/session-states'
import { isAuxiliaryWindow } from '@/store/windows'

interface QuickEntryBridgeParams {
  startFreshSessionDraft: () => void
  submitText: (text: string) => Promise<unknown> | unknown
}

// The session picker is a capture aid, not a browser — a handful of recent
// rows is the whole point. Agent options are separate and follow profile order.
const QUICK_ENTRY_SESSION_OPTIONS = 5
const QUICK_ENTRY_GROUPS_REQUEST_EVENT = 'hermes:quick-entry:groups-request'
const QUICK_ENTRY_GROUPS_CHANGED_EVENT = 'hermes:quick-entry:groups-changed'
const QUICK_ENTRY_GROUP_OPEN_EVENT = 'hermes:quick-entry:group-open'
/** Bot Mode answers with per-profile presentation (title, avatar image). */
const QUICK_ENTRY_AGENTS_REQUEST_EVENT = 'hermes:quick-entry:agents-request'
const QUICK_ENTRY_AGENTS_CHANGED_EVENT = 'hermes:quick-entry:agents-changed'
const QUICK_ENTRY_GROUP_POST_EVENT = 'hermes:quick-entry:group-post'
const QUICK_ENTRY_GROUP_FEED_REQUEST_EVENT = 'hermes:quick-entry:group-feed-request'
const QUICK_ENTRY_GROUP_FEED_CHANGED_EVENT = 'hermes:quick-entry:group-feed-changed'

function requestGroupFeed(groupId: string): HudRoomFeed | null {
  let feed: HudRoomFeed | null = null

  window.dispatchEvent(
    new CustomEvent(QUICK_ENTRY_GROUP_FEED_REQUEST_EVENT, {
      detail: {
        groupId,
        respond: (value: unknown) => {
          if (value && typeof value === 'object' && Array.isArray((value as HudRoomFeed).entries)) {
            feed = value as HudRoomFeed
          }
        }
      }
    })
  )

  return feed
}

function postGroupLine(groupId: string, text: string): boolean {
  let ok = false

  window.dispatchEvent(
    new CustomEvent(QUICK_ENTRY_GROUP_POST_EVENT, {
      detail: { groupId, text, respond: (value: unknown) => (ok = value === true) }
    })
  )

  return ok
}

/**
 * The HUD talking into a room. Main relays the HUD's requests here (the
 * room engine lives in this window), this answers through Bot Mode's events,
 * and while the HUD is watching a room every change to the room log is
 * pushed back so the HUD's panel grows as members reply.
 */
function useHudRoomRelay(): void {
  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    const api = window.hermesDesktop?.hud

    if (!api?.onRoomFeedRequest || !api.reportRoomFeed) {
      return
    }

    let watched: null | string = null

    const offFeed = api.onRoomFeedRequest(({ groupId, requestId }) => {
      api.reportRoomFeed?.({ feed: requestGroupFeed(String(groupId ?? '').trim()), requestId })
    })

    const offPost =
      api.onRoomPost?.(({ groupId, requestId, text }) => {
        api.reportRoomPost?.({ ok: postGroupLine(String(groupId ?? '').trim(), String(text ?? '')), requestId })
      }) ?? (() => {})

    const offWatch =
      api.onWatchRoom?.(({ groupId }) => {
        watched = typeof groupId === 'string' && groupId.trim() ? groupId.trim() : null
      }) ?? (() => {})

    const onChanged = () => {
      if (watched) {
        const feed = requestGroupFeed(watched)

        if (feed) {
          api.pushRoomFeed?.(feed)
        }
      }
    }

    window.addEventListener(QUICK_ENTRY_GROUP_FEED_CHANGED_EVENT, onChanged)

    return () => {
      offFeed()
      offPost()
      offWatch()
      window.removeEventListener(QUICK_ENTRY_GROUP_FEED_CHANGED_EVENT, onChanged)
    }
  }, [])
}

interface AgentDecoration {
  image?: string
  title?: string
}

function agentDecorations(): Record<string, AgentDecoration> {
  let decorations: Record<string, AgentDecoration> = {}

  window.dispatchEvent(
    new CustomEvent(QUICK_ENTRY_AGENTS_REQUEST_EVENT, {
      detail: {
        respond: (value: unknown) => {
          if (value && typeof value === 'object' && !Array.isArray(value)) {
            decorations = value as Record<string, AgentDecoration>
          }
        }
      }
    })
  )

  return decorations
}

function agentOptions(): QuickEntryAgentOption[] {
  const colors = $profileColors.get()
  const profiles = $profiles.get()

  const ordered = [
    ...profiles.filter(profile => profile.is_default),
    ...sortByProfileOrder(
      profiles.filter(profile => !profile.is_default),
      $profileOrder.get()
    )
  ]

  const decorations = agentDecorations()

  const options = ordered.map(profile => {
    const key = normalizeProfileKey(profile.name)
    const displayName = quickEntryAgentDisplayName(key, profileLabel(profile))
    const decoration = decorations[key] ?? decorations[profile.name] ?? {}
    const title = typeof decoration.title === 'string' && decoration.title.trim() ? decoration.title.trim() : undefined

    const image =
      typeof decoration.image === 'string' && decoration.image.startsWith('data:image/') ? decoration.image : undefined

    return {
      color: colors[key],
      displayName,
      emoji: agentEmoji(key, displayName, title),
      ...(image ? { image } : {}),
      profile: key,
      reachable: true,
      ...(title ? { title } : {})
    }
  })

  return options.length > 0 ? options : QUICK_ENTRY_FALLBACK_AGENTS
}

function sessionOptions(): QuickEntrySessionOption[] {
  return $sessions
    .get()
    .filter(session => !session.archived)
    .slice(0, QUICK_ENTRY_SESSION_OPTIONS)
    .map(session => ({
      id: session.id,
      title: session.title?.trim() || session.preview?.trim() || session.id
    }))
}

function groupOptions(): QuickEntryGroupOption[] {
  let groups: QuickEntryGroupOption[] = []

  window.dispatchEvent(
    new CustomEvent(QUICK_ENTRY_GROUPS_REQUEST_EVENT, {
      detail: {
        respond: (next: QuickEntryGroupOption[]) => {
          groups = Array.isArray(next) ? next : []
        }
      }
    })
  )

  return groups
}

/**
 * Wires the global-hotkey Quick Entry window back into the app, both ways:
 *
 * - **Inbound:** text captured there is routed by target and submitted through
 *   THIS window's normal prompt machinery — current chat rides `submitText`, a
 *   picked stored session rides the session-tile delegate (resume + submit,
 *   background, without touching the primary view — the same path tiled
 *   sessions use), and "new session" is a fresh draft + submit, exactly what
 *   clicking New Chat and typing does. One submit pipeline, no bespoke RPC.
 * - **Outbound:** gateway connection state + the recent-session list are pushed
 *   to the quick window (via main, which caches the latest push), so its input
 *   disables with a reconnect hint whenever the backend is unreachable.
 *
 * Handlers register ONCE through refs tracking the latest callbacks —
 * re-registering on identity churn leaves a nulled-handler window that can drop
 * a submit (the same bug shape use-pet-bridge guards). Primary window only: a
 * secondary session window must not also claim the global capture channel, or
 * one keystroke would send N prompts.
 */
export function useQuickEntryBridge({ startFreshSessionDraft, submitText }: QuickEntryBridgeParams): void {
  const submitTextRef = useRef(submitText)
  submitTextRef.current = submitText
  const startFreshRef = useRef(startFreshSessionDraft)
  startFreshRef.current = startFreshSessionDraft

  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    setQuickEntrySubmitHandler(payload => {
      if (payload.action === 'open-group') {
        const groupId = payload.groupId.trim()
        let opened = false

        window.dispatchEvent(
          new CustomEvent(QUICK_ENTRY_GROUP_OPEN_EVENT, {
            detail: {
              groupId,
              respond: (ok: boolean) => {
                opened = ok === true
              }
            }
          })
        )

        window.hermesDesktop?.quickEntry?.reportLaunchResult?.({
          ...(opened ? {} : { error: 'This group is no longer available.' }),
          ok: opened,
          profile: `group:${groupId}`,
          requestId: payload.requestId
        })

        return
      }

      if (payload.action === 'open-agent') {
        const profile = normalizeProfileKey(payload.profile)
        const profiles = $profiles.get()

        const exists =
          profiles.length === 0 || profiles.some(candidate => normalizeProfileKey(candidate.name) === profile)

        if (!exists) {
          window.hermesDesktop?.quickEntry?.reportLaunchResult?.({
            error: 'This agent is no longer available.',
            ok: false,
            profile,
            requestId: payload.requestId
          })

          return
        }

        // Resolve the exact selected profile before creating its separate HUD
        // renderer. A stale tile can therefore fail locally and leave Quick
        // Entry visible, rather than opening a disconnected lookalike.
        void window.hermesDesktop
          .getConnection(profile)
          .then(() => openHudForProfile(profile))
          .then(ok => {
            window.hermesDesktop?.quickEntry?.reportLaunchResult?.({
              ...(ok ? {} : { error: 'Hermes could not open this agent.' }),
              ok,
              profile,
              requestId: payload.requestId
            })
          })
          .catch(() => {
            window.hermesDesktop?.quickEntry?.reportLaunchResult?.({
              error: 'The selected agent is not ready yet.',
              ok: false,
              profile,
              requestId: payload.requestId
            })
          })

        return
      }

      const { target, text } = payload

      if (target === QUICK_TARGET_NEW) {
        // Same as the user clicking New Chat and typing: fresh draft, then the
        // normal submit creates the backend session.
        startFreshRef.current()
        void submitTextRef.current(text)

        return
      }

      if (target !== QUICK_TARGET_CURRENT) {
        // A picked stored session: resume + submit in the background through
        // the session-tile delegate so the primary view stays where it is.
        const delegate = sessionTileDelegate()

        if (delegate) {
          void delegate
            .resumeTile(target)
            .then(runtimeId => delegate.submitToSession(runtimeId, text))
            // A dead/undeliverable target must not swallow the prompt.
            .catch(() => void submitTextRef.current(text))

          return
        }
      }

      void submitTextRef.current(text)
    })

    const dispose = initQuickEntryBridge()

    return () => {
      setQuickEntrySubmitHandler(null)
      dispose()
    }
  }, [])

  useHudRoomRelay()

  // The HUD's room switcher is remote control: main relays "open this room"
  // here, and it goes through the same group-open event Quick Entry's tiles
  // use, so one plugin listener serves both launchers.
  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    return window.hermesDesktop?.hud?.onOpenRoom?.(({ groupId }) => {
      window.dispatchEvent(
        new CustomEvent(QUICK_ENTRY_GROUP_OPEN_EVENT, {
          detail: { groupId: String(groupId ?? '').trim(), respond: () => undefined }
        })
      )
    })
  }, [])

  // Quick Entry cannot call the gateway from its auxiliary renderer. Handle
  // its AI coaching request here, using the chosen session when available,
  // and return only a preview — never a submission.
  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    const api = window.hermesDesktop?.quickEntry

    if (!api?.onPromptCoachRequest || !api.reportPromptCoachResult) {
      return
    }

    return api.onPromptCoachRequest(request => {
      const analysis = analyzePromptDraft(request.text)

      if (!analysis) {
        api.reportPromptCoachResult?.({ analysis: null, requestId: request.requestId, text: request.text })

        return
      }

      const sessionId =
        request.target === QUICK_TARGET_CURRENT || request.target === QUICK_TARGET_NEW ? undefined : request.target

      void enhancePromptCoachWithAI(request.text, analysis, sessionId).then(enhanced => {
        api.reportPromptCoachResult?.({ analysis: enhanced, requestId: request.requestId, text: request.text })
      })
    })
  }, [])

  // Push gateway truth into the quick window whenever it changes: connection
  // state gates its input; the recent-session list feeds its target picker.
  useEffect(() => {
    if (isAuxiliaryWindow()) {
      return
    }

    const api = window.hermesDesktop?.quickEntry

    if (!api?.pushState) {
      return
    }

    const push = () => {
      api.pushState({
        agents: agentOptions(),
        connected: $gatewayState.get() === 'open',
        groups: groupOptions(),
        sessions: sessionOptions()
      })
    }

    push()

    const offGateway = $gatewayState.listen(push)
    const offSessions = $sessions.listen(push)
    const offProfiles = $profiles.listen(push)
    const offProfileOrder = $profileOrder.listen(push)
    const offProfileColors = $profileColors.listen(push)
    window.addEventListener(QUICK_ENTRY_GROUPS_CHANGED_EVENT, push)
    window.addEventListener(QUICK_ENTRY_AGENTS_CHANGED_EVENT, push)

    return () => {
      offGateway()
      offSessions()
      offProfiles()
      offProfileOrder()
      offProfileColors()
      window.removeEventListener(QUICK_ENTRY_GROUPS_CHANGED_EVENT, push)
      window.removeEventListener(QUICK_ENTRY_AGENTS_CHANGED_EVENT, push)
    }
  }, [])
}
