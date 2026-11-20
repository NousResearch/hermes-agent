import { useMemo } from 'react'

import type { CreateBackendSessionForSend } from '@/app/session/hooks/use-session-actions/create-overrides'
import { resolveComposerSessionKey } from '@/store/session'
import type { SessionInfo } from '@/types/hermes'

import { isNewChatRoute, primaryRouteSelectedSessionId, routeSessionId } from '../routes'

import type { VoiceCallTarget } from './composer/voice-call-target'
import { isRouteSessionMismatch } from './route-session-state'

interface ChatViewTargetInput {
  activeSessionId: null | string
  createVoiceDraftSession?: CreateBackendSessionForSend
  freshDraftKey: string
  freshDraftReady: boolean
  gatewaySwapTarget: unknown
  isPrimary: boolean
  messagesEmpty: boolean
  pathname: string
  selectedSessionId: null | string
  sessions: SessionInfo[]
  transcriptStoredSessionId: null | string
}

/** Which session a chat view addresses: its composer scope, its route, and its GPT-Live calls. */
export function useChatViewTargets(input: ChatViewTargetInput) {
  const { activeSessionId, gatewaySwapTarget, isPrimary, pathname, selectedSessionId, sessions } = input

  // Durable composer/queue scope (lineage root) so auto-compression tip rotation
  // does not wipe an in-progress draft or orphan /queue entries. For the
  // primary view, the route is authoritative over the store selection — the
  // latter can be momentarily null/stale mid-switch, which used to leak into
  // the composer's scope key (#59305). A tile has no route, so it always uses
  // its own selection directly.
  const queueSessionKey = useMemo(() => {
    const effectiveSelectedSessionId = isPrimary
      ? primaryRouteSelectedSessionId(pathname, selectedSessionId)
      : selectedSessionId

    return resolveComposerSessionKey(effectiveSelectedSessionId, sessions)
  }, [isPrimary, pathname, selectedSessionId, sessions])

  // A tile IS its session — no route involved, never "mismatched".
  const routedSessionId = isPrimary ? routeSessionId(pathname) : selectedSessionId

  // The URL points at a session the store hasn't loaded yet (sidebar / cmd-K /
  // direct nav). Derived in render so the swap reads instantly: the same frame
  // the id changes we drop the old transcript and show the loader, instead of
  // waiting for the resume effect (which paints a frame later) to clear them.
  const routeSessionMismatch = isPrimary
    ? isRouteSessionMismatch(routedSessionId, selectedSessionId, sessions, {
        activeRuntimeId: activeSessionId,
        contextSwitching: Boolean(gatewaySwapTarget),
        messagesEmpty: input.messagesEmpty,
        transcriptStoredSessionId: input.transcriptStoredSessionId
      })
    : false

  return {
    isRoutedSessionView: Boolean(routedSessionId),
    queueSessionKey,
    routedSessionId,
    routeSessionMismatch,
    voiceCall: voiceCallTarget(input, routedSessionId)
  }
}

/**
 * Where a GPT-Live call started from this view records: the selected stored chat,
 * only once selection, route and transcript agree (the transcript-retention fallback
 * can display a chat while selection catches up, and capture must not bind to a stale
 * selection), or a New Chat draft the call may create.
 */
function voiceCallTarget(input: ChatViewTargetInput, routedSessionId: null | string): VoiceCallTarget {
  const { activeSessionId, gatewaySwapTarget, isPrimary, selectedSessionId, transcriptStoredSessionId } = input

  const routeAgrees =
    !isPrimary ||
    (primaryRouteSelectedSessionId(input.pathname, selectedSessionId) &&
      !isRouteSessionMismatch(routedSessionId, selectedSessionId, input.sessions))

  const bound =
    Boolean(activeSessionId) &&
    !gatewaySwapTarget &&
    routeAgrees &&
    (!transcriptStoredSessionId || transcriptStoredSessionId === selectedSessionId)

  const onNewChatRoute = isPrimary && isNewChatRoute(input.pathname)
  const canCreateDraft = onNewChatRoute && input.freshDraftReady && !gatewaySwapTarget && !activeSessionId && !selectedSessionId

  return {
    createDraftSession: canCreateDraft ? input.createVoiceDraftSession : undefined,
    draftKey: onNewChatRoute ? input.freshDraftKey : undefined,
    routeSessionId: routedSessionId,
    storedSessionId: bound ? selectedSessionId : null
  }
}
