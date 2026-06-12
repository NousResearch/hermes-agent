import type { MutableRefObject } from 'react'
import { useCallback, useRef } from 'react'
import type { NavigateFunction } from 'react-router-dom'

import { deleteSession, getSessionMessages, listAllProfileSessions, setSessionArchived } from '@/hermes'
import { useI18n } from '@/i18n'
import { type ChatMessage, chatMessageText, preserveLocalAssistantErrors, toChatMessages } from '@/lib/chat-messages'
import { normalizePersonalityValue } from '@/lib/chat-runtime'
import { embeddedImageUrls, textWithoutEmbeddedImages } from '@/lib/embedded-images'
import { setSessionYolo } from '@/lib/yolo-session'
import { clearQueuedPrompts } from '@/store/composer-queue'
import { $pinnedSessionIds } from '@/store/layout'
import { clearNotifications, notify, notifyError } from '@/store/notifications'
import { requestDesktopOnboarding } from '@/store/onboarding'
import { $activeGatewayProfile, $newChatProfile, ensureGatewayProfile, normalizeProfileKey } from '@/store/profile'
import {
  $currentCwd,
  $messages,
  $sessions,
  $yoloActive,
  sessionPinId,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setCurrentBranch,
  setCurrentCwd,
  setCurrentFastMode,
  setCurrentModel,
  setCurrentPersonality,
  setCurrentProvider,
  setCurrentReasoningEffort,
  setCurrentServiceTier,
  setCurrentUsage,
  setFreshDraftReady,
  setIntroSeed,
  setMessages,
  setSelectedStoredSessionId,
  setSessions,
  setSessionStartedAt,
  setSessionsTotal,
  setTurnStartedAt,
  setYoloActive,
  workspaceCwdForNewSession
} from '@/store/session'
import { reportBackendContract } from '@/store/updates'
import { isWatchWindow } from '@/store/windows'
import type { SessionCreateResponse, SessionInfo, SessionResumeResponse, SessionRuntimeInfo, UsageStats } from '@/types/hermes'

import { NEW_CHAT_ROUTE, sessionRoute, SETTINGS_ROUTE } from '../../routes'
import type { ClientSessionState, SidebarNavItem } from '../../types'

interface SessionActionsOptions {
  activeSessionId: string | null
  activeSessionIdRef: MutableRefObject<string | null>
  busyRef: MutableRefObject<boolean>
  creatingSessionRef: MutableRefObject<boolean>
  ensureSessionState: (sessionId: string, storedSessionId?: string | null) => ClientSessionState
  getRouteToken: () => string
  navigate: NavigateFunction
  requestGateway: <T>(method: string, params?: Record<string, unknown>) => Promise<T>
  runtimeIdByStoredSessionIdRef: MutableRefObject<Map<string, string>>
  selectedStoredSessionId: string | null
  selectedStoredSessionIdRef: MutableRefObject<string | null>
  sessionStateByRuntimeIdRef: MutableRefObject<Map<string, ClientSessionState>>
  syncSessionStateToView: (sessionId: string, state: ClientSessionState) => void
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: string | null
  ) => ClientSessionState
}

function withAppendedText(message: ChatMessage, suffix: string): ChatMessage {
  let appended = false

  const parts = message.parts.map(part => {
    if (part.type !== 'text' || appended) {
      return part
    }

    appended = true

    return { ...part, text: `${part.text}${suffix}` }
  })

  return appended ? { ...message, parts } : message
}

function preserveReasoningParts(message: ChatMessage, previous: ChatMessage): ChatMessage {
  if (message.parts.some(part => part.type === 'reasoning')) {
    return message
  }

  const reasoningParts = previous.parts.filter(part => part.type === 'reasoning')

  return reasoningParts.length ? { ...message, parts: [...reasoningParts, ...message.parts] } : message
}

function chatMessagesEquivalent(a: ChatMessage, b: ChatMessage): boolean {
  if (
    a.id !== b.id ||
    a.role !== b.role ||
    a.pending !== b.pending ||
    a.error !== b.error ||
    a.hidden !== b.hidden ||
    a.branchGroupId !== b.branchGroupId
  ) {
    return false
  }

  if (a.parts.length !== b.parts.length) {
    return false
  }

  return a.parts.every((part, index) => JSON.stringify(part) === JSON.stringify(b.parts[index]))
}

function chatMessageArraysEquivalent(a: ChatMessage[], b: ChatMessage[]): boolean {
  return a.length === b.length && a.every((message, index) => chatMessagesEquivalent(message, b[index]))
}

function reconcileResumeMessages(nextMessages: ChatMessage[], previousMessages: ChatMessage[]): ChatMessage[] {
  if (!previousMessages.length) {
    return nextMessages
  }

  const previousByRoleOrdinal = new Map<string, ChatMessage>()
  const previousRoleCounts = new Map<string, number>()

  for (const message of previousMessages) {
    const ordinal = previousRoleCounts.get(message.role) ?? 0
    previousRoleCounts.set(message.role, ordinal + 1)
    previousByRoleOrdinal.set(`${message.role}:${ordinal}`, message)
  }

  const nextRoleCounts = new Map<string, number>()

  return nextMessages.map(message => {
    const ordinal = nextRoleCounts.get(message.role) ?? 0
    nextRoleCounts.set(message.role, ordinal + 1)

    const previous = previousByRoleOrdinal.get(`${message.role}:${ordinal}`)

    if (!previous) {
      return message
    }

    const nextText = chatMessageText(message).trim()
    const previousText = chatMessageText(previous)
    const previousVisibleText = textWithoutEmbeddedImages(previousText)
    let preserved = message

    if (nextText === previousVisibleText || nextText === previousText.trim()) {
      preserved = preserveReasoningParts(preserved, previous)
    }

    const previousImages = embeddedImageUrls(previousText)

    if (!previousImages.length || embeddedImageUrls(chatMessageText(preserved)).length) {
      return preserved
    }

    if (nextText !== previousVisibleText) {
      return preserved
    }

    return withAppendedText(preserved, previousImages.map(url => `\n${url}`).join(''))
  })
}

function upsertOptimisticSession(
  created: SessionCreateResponse,
  id: string,
  title: string | null = null,
  preview: string | null = null
) {
  const now = Date.now() / 1000
  // Stamp the profile the session was just created on (= the live gateway's
  // profile) so the scoped sidebar shows the new row immediately instead of
  // filtering it out as "default" until the aggregator re-fetches.
  const profileKey = normalizeProfileKey($activeGatewayProfile.get())

  const session: SessionInfo = {
    cwd: created.info?.cwd ?? null,
    ended_at: null,
    id,
    input_tokens: 0,
    is_active: true,
    is_default_profile: profileKey === 'default',
    last_active: now,
    message_count: created.message_count ?? created.messages?.length ?? 0,
    model: created.info?.model ?? null,
    output_tokens: 0,
    preview,
    profile: profileKey,
    source: 'tui',
    started_at: now,
    title,
    tool_call_count: 0
  }

  setSessions(prev => [session, ...prev.filter(s => s.id !== id)])
}

function patchSessionWorkspace(sessionId: string, cwd: string | undefined) {
  if (!cwd) {
    return
  }

  setSessions(prev => prev.map(session => (session.id === sessionId ? { ...session, cwd } : session)))
}

function sessionMatchesStoredId(session: SessionInfo, storedSessionId: string): boolean {
  return session.id === storedSessionId || session._lineage_root_id === storedSessionId
}

function upsertResolvedSession(session: SessionInfo, storedSessionId: string) {
  const lineage = session._lineage_root_id ?? session.id

  setSessions(prev => [
    session,
    ...prev.filter(existing => {
      if (sessionMatchesStoredId(existing, storedSessionId)) {
        return false
      }

      return (existing._lineage_root_id ?? existing.id) !== lineage
    })
  ])
}

async function resolveStoredSession(storedSessionId: string): Promise<SessionInfo | undefined> {
  const cached = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

  if (cached) {
    return cached
  }

  try {
    const result = await listAllProfileSessions(500, 0, 'include', 'recent', 'all')
    const resolved = result.sessions.find(session => sessionMatchesStoredId(session, storedSessionId))

    if (resolved) {
      upsertResolvedSession(resolved, storedSessionId)
    }

    return resolved
  } catch {
    return undefined
  }
}

type SessionRuntimeStatePatch = Partial<
  Pick<
    ClientSessionState,
    | 'branch'
    | 'cwd'
    | 'fast'
    | 'model'
    | 'personality'
    | 'provider'
    | 'reasoningEffort'
    | 'serviceTier'
    | 'yolo'
  >
>

function applyRuntimeInfo(info: SessionRuntimeInfo | undefined): SessionRuntimeStatePatch | null {
  if (!info) {
    return null
  }

  const sessionState: SessionRuntimeStatePatch = {}

  reportBackendContract(info.desktop_contract)

  if (info.credential_warning) {
    requestDesktopOnboarding(info.credential_warning)
  }

  if (typeof info.model === 'string') {
    setCurrentModel(info.model)
    sessionState.model = info.model
  }

  if (typeof info.provider === 'string') {
    setCurrentProvider(info.provider)
    sessionState.provider = info.provider
  }

  if (info.cwd) {
    setCurrentCwd(info.cwd)
    sessionState.cwd = info.cwd
  }

  if (info.branch !== undefined) {
    setCurrentBranch(info.branch || '')
    sessionState.branch = info.branch || ''
  }

  if (typeof info.personality === 'string') {
    const personality = normalizePersonalityValue(info.personality)
    setCurrentPersonality(personality)
    sessionState.personality = personality
  }

  if (typeof info.reasoning_effort === 'string') {
    setCurrentReasoningEffort(info.reasoning_effort)
    sessionState.reasoningEffort = info.reasoning_effort
  }

  if (typeof info.service_tier === 'string') {
    setCurrentServiceTier(info.service_tier)
    sessionState.serviceTier = info.service_tier
  }

  if (typeof info.fast === 'boolean') {
    setCurrentFastMode(info.fast)
    sessionState.fast = info.fast
  }

  if (typeof info.yolo === 'boolean') {
    setYoloActive(info.yolo)
    sessionState.yolo = info.yolo
  }

  if (info.usage) {
    setCurrentUsage(current => ({ ...current, ...info.usage }))
  }

  return sessionState
}

function applyStoredSessionPreviewRuntimeInfo(stored: { model?: null | string } | undefined) {
  setCurrentModel(stored?.model || '')
  setCurrentProvider('')
  setCurrentReasoningEffort('')
  setCurrentServiceTier('')
  setCurrentFastMode(false)
  setYoloActive(false)
  setCurrentPersonality('')
}

export function useSessionActions({
  activeSessionId,
  activeSessionIdRef,
  busyRef,
  creatingSessionRef,
  ensureSessionState,
  getRouteToken,
  navigate,
  requestGateway,
  runtimeIdByStoredSessionIdRef,
  selectedStoredSessionId,
  selectedStoredSessionIdRef,
  sessionStateByRuntimeIdRef,
  syncSessionStateToView,
  updateSessionState
}: SessionActionsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const resumeRequestRef = useRef(0)

  const startFreshSessionDraft = useCallback(
    (replaceRoute = false) => {
      busyRef.current = false
      setBusy(false)
      setAwaitingResponse(false)
      clearNotifications()
      setIntroSeed(seed => seed + 1)
      navigate(NEW_CHAT_ROUTE, { replace: replaceRoute })
      setActiveSessionId(null)
      activeSessionIdRef.current = null
      setSelectedStoredSessionId(null)
      selectedStoredSessionIdRef.current = null
      setMessages([])
      setCurrentUsage({
        calls: 0,
        input: 0,
        output: 0,
        total: 0
      })
      setSessionStartedAt(null)
      setTurnStartedAt(null)
      // New chats start in the configured default project dir when set,
      // otherwise the sticky last-used workspace (PR #37586).
      setCurrentModel('')
      setCurrentProvider('')
      setCurrentReasoningEffort('')
      setCurrentServiceTier('')
      setCurrentFastMode(false)
      setYoloActive(false)
      setCurrentCwd(workspaceCwdForNewSession())
      setCurrentBranch('')
      // Never clear the composer here — ChatBar's per-thread draft swap owns it.
      setFreshDraftReady(true)
    },
    [activeSessionIdRef, busyRef, navigate, selectedStoredSessionIdRef]
  )

  const createBackendSessionForSend = useCallback(
    async (preview: string | null = null): Promise<string | null> => {
      const startingActiveSessionId = activeSessionIdRef.current
      const startingStoredSessionId = selectedStoredSessionIdRef.current
      const startingRouteToken = getRouteToken()

      creatingSessionRef.current = true

      try {
        // A plain new session (top "New Session", /new, keybind) leaves
        // $newChatProfile null to mean "use the live context"; the per-profile
        // "+" sets it explicitly. Resolve null to the active gateway profile so
        // session.create always carries it: in global-remote mode one backend
        // serves every profile, so an omitted profile param silently lands the
        // chat on the launch (default) profile — the "rubberbands back to
        // default" bug. This is a no-op for single-profile/local-pooled users:
        // a backend resolves its own launch profile to None (_profile_home).
        const newChatProfile = $newChatProfile.get() ?? normalizeProfileKey($activeGatewayProfile.get())
        await ensureGatewayProfile(newChatProfile)
        const cwd = $currentCwd.get().trim() || workspaceCwdForNewSession()

        const created = await requestGateway<SessionCreateResponse>('session.create', {
          cols: 96,
          ...(cwd && { cwd }),
          ...(newChatProfile ? { profile: newChatProfile } : {})
        })

        const stored = created.stored_session_id ?? null

        if (
          activeSessionIdRef.current !== startingActiveSessionId ||
          selectedStoredSessionIdRef.current !== startingStoredSessionId ||
          getRouteToken() !== startingRouteToken
        ) {
          await requestGateway('session.close', { session_id: created.session_id }).catch(() => undefined)

          return null
        }

        activeSessionIdRef.current = created.session_id
        selectedStoredSessionIdRef.current = stored
        ensureSessionState(created.session_id, stored)

        if (stored) {
          // Seed the sidebar preview with the user's first message so the row
          // reads meaningfully while the turn is in flight, instead of flashing
          // "Untitled session" until the turn persists and auto-title runs. The
          // server later returns its own preview/title and supersedes this.
          upsertOptimisticSession(created, stored, null, preview?.trim() || null)
          navigate(sessionRoute(stored), { replace: true })
        }

        setFreshDraftReady(false)
        setActiveSessionId(created.session_id)
        setSelectedStoredSessionId(stored)
        setSessionStartedAt(Date.now())
        const yoloArmed = $yoloActive.get()
        const runtimeInfo = applyRuntimeInfo(created.info)

        if (runtimeInfo) {
          updateSessionState(created.session_id, state => ({ ...state, ...runtimeInfo }), stored)
        }

        // User may have armed YOLO on the new-chat draft before the runtime
        // session existed — apply it to the freshly created session.
        if (yoloArmed) {
          await setSessionYolo(requestGateway, created.session_id, true).catch(() => undefined)
        }

        return created.session_id
      } finally {
        window.setTimeout(() => {
          creatingSessionRef.current = false
        }, 0)
      }
    },
    [
      activeSessionIdRef,
      creatingSessionRef,
      ensureSessionState,
      getRouteToken,
      navigate,
      requestGateway,
      selectedStoredSessionIdRef,
      updateSessionState
    ]
  )

  const selectSidebarItem = useCallback(
    (item: SidebarNavItem) => {
      if (item.action === 'new-session') {
        startFreshSessionDraft()

        return
      }

      if (item.route) {
        navigate(item.route)
      }
    },
    [navigate, startFreshSessionDraft]
  )

  const openSettings = useCallback(() => {
    navigate(SETTINGS_ROUTE)
  }, [navigate])

  const closeSettings = useCallback(() => {
    if (selectedStoredSessionId) {
      navigate(sessionRoute(selectedStoredSessionId))

      return
    }

    navigate(NEW_CHAT_ROUTE)
  }, [navigate, selectedStoredSessionId])

  const resumeSession = useCallback(
    async (storedSessionId: string, replaceRoute = false) => {
      const requestId = resumeRequestRef.current + 1
      resumeRequestRef.current = requestId

      const isCurrentResume = () =>
        resumeRequestRef.current === requestId && selectedStoredSessionIdRef.current === storedSessionId

      // Swap the single live gateway to this session's profile before any
      // gateway call (no-op when it's already on that profile / single-profile).
      const storedForProfile = await resolveStoredSession(storedSessionId)
      const sessionProfile = storedForProfile?.profile

      if (resumeRequestRef.current !== requestId) {
        return
      }

      await ensureGatewayProfile(sessionProfile)

      const cachedRuntimeId = runtimeIdByStoredSessionIdRef.current.get(storedSessionId)
      const cachedState = cachedRuntimeId && sessionStateByRuntimeIdRef.current.get(cachedRuntimeId)

      if (cachedRuntimeId && cachedState) {
        const stored = $sessions.get().find(session => session.id === storedSessionId)

        const cachedViewState =
          !cachedState.model && stored?.model != null
            ? {
                ...cachedState,
                model: stored.model || ''
              }
            : cachedState

        if (cachedViewState !== cachedState) {
          sessionStateByRuntimeIdRef.current.set(cachedRuntimeId, cachedViewState)
        }

        setFreshDraftReady(false)
        clearNotifications()
        setSelectedStoredSessionId(storedSessionId)
        selectedStoredSessionIdRef.current = storedSessionId
        setActiveSessionId(cachedRuntimeId)
        activeSessionIdRef.current = cachedRuntimeId
        syncSessionStateToView(cachedRuntimeId, cachedViewState)
        setCurrentCwd(cachedViewState.cwd)
        setCurrentBranch(cachedViewState.branch)
        setSessionStartedAt(Date.now())

        try {
          const usage = await requestGateway<UsageStats>('session.usage', { session_id: cachedRuntimeId })

          if (!isCurrentResume()) {
            return
          }

          if (usage) {
            setCurrentUsage(current => ({ ...current, ...usage }))
          }

          return
        } catch {
          // The cached runtime id was minted by a prior backend instance. A
          // pooled profile backend that gets idle-reaped (pruneSecondaryGateways)
          // and respawned across a profile swap mints fresh ids, so this mapping
          // now 404s ("session not found"). Drop it and fall through to a full
          // resume that rebinds a live runtime id.
          if (!isCurrentResume()) {
            return
          }

          runtimeIdByStoredSessionIdRef.current.delete(storedSessionId)
          sessionStateByRuntimeIdRef.current.delete(cachedRuntimeId)
        }
      }

      setFreshDraftReady(false)
      setActiveSessionId(null)
      activeSessionIdRef.current = null
      busyRef.current = true
      setBusy(true)
      setAwaitingResponse(false)
      clearNotifications()
      setSelectedStoredSessionId(storedSessionId)
      selectedStoredSessionIdRef.current = storedSessionId
      setSessionStartedAt(Date.now())
      const stored = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
      applyStoredSessionPreviewRuntimeInfo(stored)

      if (stored) {
        setCurrentUsage(current => ({
          ...current,
          input: stored.input_tokens || 0,
          output: stored.output_tokens || 0,
          total: (stored.input_tokens || 0) + (stored.output_tokens || 0)
        }))
      }

      let resumedRunning = false

      try {
        const watchWindow = isWatchWindow()
        let localSnapshot = $messages.get()

        try {
          // Watch windows skip REST prefetch — lazy resume attaches the live mirror.
          if (!watchWindow) {
            const storedMessages = await getSessionMessages(storedSessionId, sessionProfile)

            if (isCurrentResume()) {
              localSnapshot = preserveLocalAssistantErrors(toChatMessages(storedMessages.messages), $messages.get())

              if (!chatMessageArraysEquivalent($messages.get(), localSnapshot)) {
                setMessages(localSnapshot)
              }
            }
          }
        } catch {
          // Non-fatal: gateway resume below can still hydrate the session.
        }

        const resumed = await requestGateway<SessionResumeResponse>('session.resume', {
          session_id: storedSessionId,
          cols: 96,
          ...(watchWindow ? { lazy: true } : {}),
          ...(sessionProfile ? { profile: sessionProfile } : {})
        })

        if (!isCurrentResume()) {
          return
        }

        const currentMessages = $messages.get()

        const resumedMessages = preserveLocalAssistantErrors(
          reconcileResumeMessages(toChatMessages(resumed.messages), currentMessages),
          currentMessages
        )
        // Keep the local snapshot when resume would only reshuffle runtime projection.
        const preferredMessages =
          localSnapshot.length > 0
            ? localSnapshot
            : chatMessageArraysEquivalent(currentMessages, resumedMessages)
              ? currentMessages
              : resumedMessages

        const messagesForView = preserveLocalAssistantErrors(preferredMessages, currentMessages)

        setActiveSessionId(resumed.session_id)
        activeSessionIdRef.current = resumed.session_id
        const runtimeInfo = applyRuntimeInfo(resumed.info)

        patchSessionWorkspace(storedSessionId, runtimeInfo?.cwd)

        resumedRunning = Boolean((resumed as { running?: boolean }).running)

        updateSessionState(
          resumed.session_id,
          state => ({
            ...state,
            ...(runtimeInfo ?? {}),
            messages: messagesForView,
            busy: resumedRunning,
            awaitingResponse: resumedRunning
          }),
          storedSessionId
        )
      } catch (err) {
        if (!isCurrentResume()) {
          return
        }

        const fallback = await getSessionMessages(storedSessionId, sessionProfile)

        if (!isCurrentResume()) {
          return
        }

        setMessages(preserveLocalAssistantErrors(toChatMessages(fallback.messages), $messages.get()))
        notifyError(err, copy.resumeFailed)
      } finally {
        if (isCurrentResume()) {
          busyRef.current = resumedRunning
          setBusy(resumedRunning)
          setAwaitingResponse(resumedRunning)
        }
      }
    },
    [
      activeSessionIdRef,
      busyRef,
      copy,
      requestGateway,
      runtimeIdByStoredSessionIdRef,
      selectedStoredSessionIdRef,
      sessionStateByRuntimeIdRef,
      syncSessionStateToView,
      updateSessionState
    ]
  )

  const branchCurrentSession = useCallback(
    async (messageId?: string): Promise<boolean> => {
      const sourceSessionId = activeSessionIdRef.current

      if (!sourceSessionId) {
        notify({
          kind: 'warning',
          title: copy.nothingToBranch,
          message: copy.branchNeedsChat
        })

        return false
      }

      if (busyRef.current) {
        notify({
          kind: 'warning',
          title: copy.sessionBusy,
          message: copy.branchStopCurrent
        })

        return false
      }

      creatingSessionRef.current = true

      try {
        const currentMessages = $messages.get()

        const targetIndex = messageId
          ? currentMessages.findIndex(message => message.id === messageId)
          : currentMessages.findLastIndex(message => message.role === 'assistant' || message.role === 'user')

        const branchStart = targetIndex >= 0 ? targetIndex : Math.max(currentMessages.length - 1, 0)
        const branchEnd = targetIndex >= 0 ? targetIndex + 1 : currentMessages.length

        const branchMessages = currentMessages
          .slice(branchStart, branchEnd)
          .map(message => ({
            content: chatMessageText(message),
            source: message,
            role: message.role
          }))
          .filter(message => message.content.trim() && ['assistant', 'user'].includes(message.role))

        if (!branchMessages.length) {
          notify({
            kind: 'warning',
            title: copy.nothingToBranch,
            message: copy.branchNoText
          })

          return false
        }

        clearNotifications()

        const cwd = $currentCwd.get().trim()

        const branched = await requestGateway<SessionCreateResponse>('session.create', {
          cols: 96,
          ...(cwd && { cwd }),
          messages: branchMessages.map(({ content, role }) => ({ content, role })),
          title: copy.branchTitle
        })

        const routedSessionId = branched.stored_session_id ?? branched.session_id
        const preview = branchMessages.map(({ content }) => content).find(Boolean) ?? null

        setFreshDraftReady(false)
        upsertOptimisticSession(branched, routedSessionId, copy.branchTitle, preview)
        ensureSessionState(branched.session_id, routedSessionId)
        setActiveSessionId(branched.session_id)
        activeSessionIdRef.current = branched.session_id
        updateSessionState(
          branched.session_id,
          state => ({
            ...state,
            messages: branchMessages.map(({ source }) => source),
            busy: false,
            awaitingResponse: false
          }),
          routedSessionId
        )
        setSelectedStoredSessionId(routedSessionId)
        selectedStoredSessionIdRef.current = routedSessionId
        navigate(sessionRoute(routedSessionId))

        const runtimeInfo = applyRuntimeInfo(branched.info)

        patchSessionWorkspace(routedSessionId, runtimeInfo?.cwd)

        if (runtimeInfo) {
          updateSessionState(branched.session_id, state => ({ ...state, ...runtimeInfo }), routedSessionId)
        }

        return true
      } catch (err) {
        notifyError(err, copy.branchFailed)

        return false
      } finally {
        window.setTimeout(() => {
          creatingSessionRef.current = false
        }, 0)
      }
    },
    [
      activeSessionIdRef,
      busyRef,
      copy,
      creatingSessionRef,
      ensureSessionState,
      navigate,
      requestGateway,
      selectedStoredSessionIdRef,
      updateSessionState
    ]
  )

  const removeSession = useCallback(
    async (storedSessionId: string) => {
      clearNotifications()

      const removed = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
      const wasSelected = selectedStoredSessionId === storedSessionId
      const closingRuntimeId = wasSelected ? activeSessionId : null
      const previousMessages = $messages.get()
      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const removedPinId = removed ? sessionPinId(removed) : storedSessionId

      setSessions(prev => prev.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
      // Keep $sessionsTotal in sync so the sidebar's "Load N more" footer
      // doesn't keep claiming the removed row is still on the server.
      setSessionsTotal(prev => Math.max(0, prev - 1))
      $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== removedPinId))

      // Tear down before awaiting so the route effect can't resume the
      // doomed session via the stale /<sid> URL.
      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        if (closingRuntimeId) {
          await requestGateway('session.close', { session_id: closingRuntimeId }).catch(() => undefined)
        }

        await deleteSession(storedSessionId, removed?.profile)
        clearQueuedPrompts(storedSessionId)

        if (closingRuntimeId) {
          clearQueuedPrompts(closingRuntimeId)
        }
      } catch (err) {
        if (removed) {
          setSessions(prev => [removed, ...prev])
          setSessionsTotal(prev => prev + 1)
        }

        $pinnedSessionIds.set(previousPinned)

        if (wasSelected) {
          setFreshDraftReady(false)
          setSelectedStoredSessionId(storedSessionId)
          selectedStoredSessionIdRef.current = storedSessionId
          const stored = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))

          if (stored) {
            setCurrentUsage(current => ({
              ...current,
              input: stored.input_tokens || 0,
              output: stored.output_tokens || 0,
              total: (stored.input_tokens || 0) + (stored.output_tokens || 0)
            }))
          }

          setMessages(previousMessages)
          navigate(sessionRoute(storedSessionId), { replace: true })

          if (closingRuntimeId) {
            setActiveSessionId(closingRuntimeId)
            activeSessionIdRef.current = closingRuntimeId
          }
        }

        notifyError(err, copy.deleteFailed)
      }
    },
    [
      activeSessionId,
      activeSessionIdRef,
      copy,
      navigate,
      requestGateway,
      selectedStoredSessionId,
      selectedStoredSessionIdRef,
      startFreshSessionDraft
    ]
  )

  const archiveSession = useCallback(
    async (storedSessionId: string) => {
      clearNotifications()

      const archived = $sessions.get().find(session => sessionMatchesStoredId(session, storedSessionId))
      const wasSelected = selectedStoredSessionId === storedSessionId
      const previousPinned = $pinnedSessionIds.get()
      // Pins are keyed on the durable lineage-root id; the stored id may be the
      // live tip after compression. Drop both so the pin can't linger.
      const archivedPinId = archived ? sessionPinId(archived) : storedSessionId

      // Soft-hide: drop from the sidebar immediately, keep the data.
      setSessions(prev => prev.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
      // Archived sessions are hidden by the listSessions(min_messages=1) query
      // on the next refresh, so they count as "removed" for the load-more
      // footer math.
      setSessionsTotal(prev => Math.max(0, prev - 1))
      $pinnedSessionIds.set(previousPinned.filter(id => id !== storedSessionId && id !== archivedPinId))

      if (wasSelected) {
        startFreshSessionDraft(true)
      }

      try {
        await setSessionArchived(storedSessionId, true, archived?.profile)
        // A sidebar refresh can race the optimistic removal while the PATCH is
        // in flight and briefly reinsert the still-unarchived backend row. Win
        // that race after the mutation succeeds so right-click → Archive does
        // not appear to do nothing until the next full refresh.
        setSessions(prev => prev.filter(session => !sessionMatchesStoredId(session, storedSessionId)))
        $pinnedSessionIds.set($pinnedSessionIds.get().filter(id => id !== storedSessionId && id !== archivedPinId))
        notify({ durationMs: 2_000, kind: 'success', message: copy.archived })
      } catch (err) {
        if (archived) {
          setSessions(prev => [archived, ...prev.filter(session => !sessionMatchesStoredId(session, storedSessionId))])
          setSessionsTotal(prev => prev + 1)
        }

        $pinnedSessionIds.set(previousPinned)
        notifyError(err, copy.archiveFailed)
      }
    },
    [copy, selectedStoredSessionId, startFreshSessionDraft]
  )

  return {
    archiveSession,
    branchCurrentSession,
    closeSettings,
    createBackendSessionForSend,
    openSettings,
    removeSession,
    resumeSession,
    selectSidebarItem,
    startFreshSessionDraft
  }
}
