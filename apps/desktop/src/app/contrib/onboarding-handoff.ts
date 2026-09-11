/** Welcome-chat creation, durable first-build handoff, and progress check-ins. */

import { useStore } from '@nanostores/react'
import { useCallback, useEffect } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/api/client'
import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import type { SessionCreateOverrides } from '@/app/session/hooks/use-session-actions/create-overrides'
import type { ClientSessionState } from '@/app/types'
import {
  $chatOnboardingThreadIds,
  endChatOnboardingSolo,
  pickOnboardingGreeting,
  startChatOnboardingSolo
} from '@/components/onboarding-chat/assembly'
import { $setupCheckIn, watchFirstBuild } from '@/components/onboarding-chat/first-build'
import {
  $setupHandoff,
  $setupSession,
  buildFirstTaskSeedMessages,
  buildHandoffCompleteNote,
  ensureSetupProfile,
  firstTaskTitle,
  markSetupHandoffDone,
  SETUP_CHAT_TITLE,
  SETUP_PROFILE
} from '@/components/onboarding-chat/setup-profile'
import { declinedLookAround, showProfileSignpost } from '@/components/onboarding-chat/signpost'
import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, activateTreePane } from '@/components/pane-shell/tree/store'
import { toChatMessages } from '@/lib/chat-messages'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { activeGatewayConnectionId, requestGatewayForAgent, requestGatewayForProfile } from '@/store/gateway'
import { loadMachineProfile } from '@/store/machine'
import { dismissNotification, notify } from '@/store/notifications'
import { $onboardingAnswers } from '@/store/onboarding-answers'
import { beginOnboardingHandoff, completeOnboardingFlow, skipGuide } from '@/store/onboarding-gate'
import { buildChatOnboardingSeedMessages } from '@/store/onboarding-script'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  ensureGatewayAgent,
  ensureGatewayProfile
} from '@/store/profile'
import {
  $activeSessionId,
  $messages,
  $selectedStoredSessionId,
  forgetSessionOwnerHintsForSession,
  getSessionOwnerHint,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setSessionOwnerHint
} from '@/store/session'
import { patchSessionTile } from '@/store/session-states'

import { BUILD_PROFILE, type HandoffDeps, type HandoffReceipt, paintHandoffBrief, startHandoff } from './handoff-leg'
import {
  $handoffError,
  handoffReceiptKey,
  readHandoffReceipt,
  retrySetupHandoff,
  saveHandoffReceipt
} from './handoff-receipt'
import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

type SeedMessage = ReturnType<typeof buildChatOnboardingSeedMessages>[number]

interface SetupStatus {
  ready?: boolean
  provider_configured?: boolean
  free_tier?: boolean
}

/** The welcome chat's owning source: its recorded owner hint, else the active
 *  gateway's registry id, else null. Null is the ambient route for the profile
 *  (a local-only install creates the setup profile as a registry secondary
 *  with no connection id and records no hint; a legacy primary has no registry
 *  id either), the same socket the welcome chat itself runs on. It is never a
 *  missing owner, and never 'local': an explicit local id would retarget a
 *  legacy remote primary onto this machine. */
function guideSourceConnectionId(guideStoredId: null | string | undefined): null | string {
  return (guideStoredId && getSessionOwnerHint(guideStoredId)?.connectionId) || activeGatewayConnectionId() || null
}

export interface OnboardingHandoffOptions {
  activeSessionIdRef: { current: null | string }
  ensureSessionState: (sessionId: string, storedSessionId?: null | string) => ClientSessionState
  updateSessionState: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId?: null | string
  ) => ClientSessionState
  createBackendSessionForSend: (
    preview?: null | string,
    seedMessages?: SeedMessage[],
    overrides?: SessionCreateOverrides
  ) => Promise<null | string>
  requestGateway: AmbientGatewayRequest
  resumeSession: ReturnType<typeof useSessionActions>['resumeSession']
  /** Runs `create` with the session-create leg pinned to `profile` instead of
   *  the selected chat's owner. The caller owns the mechanism (its own
   *  `requestGateway` is what reads the pin); the hook only needs to say which
   *  backend the new session belongs on. */
  runCreatePinnedTo: <T>(profile: string, create: () => Promise<T>) => Promise<T>
}

interface GuideSession {
  id: string
  resolved_id?: string
}

async function adoptGuideSession(
  canonical: GuideSession,
  freeTier: SetupStatus['free_tier'],
  resumeSession: OnboardingHandoffOptions['resumeSession'],
  guideRequest: AmbientGatewayRequest
): Promise<void> {
  await resumeSession(canonical.resolved_id ?? canonical.id, true)
  const adoptedRuntimeId = $activeSessionId.get()
  $chatOnboardingThreadIds.set(adoptedRuntimeId ? [canonical.id, adoptedRuntimeId] : [canonical.id])
  $setupSession.set({
    connectionId: guideSourceConnectionId(canonical.id),
    profile: SETUP_PROFILE,
    runtimeId: adoptedRuntimeId ?? canonical.id,
    storedId: canonical.id
  })

  if (freeTier) {
    await guideRequest('config.set', {
      session_id: adoptedRuntimeId ?? canonical.id,
      key: 'reasoning',
      value: 'minimal'
    })
  }
}

/** Returns the first-chat kickoff; wires the handoff and check-in effects. */
export function useOnboardingHandoff({
  activeSessionIdRef,
  ensureSessionState,
  updateSessionState,
  createBackendSessionForSend,
  requestGateway,
  resumeSession,
  runCreatePinnedTo
}: OnboardingHandoffOptions) {
  const kickoffFirstChat = useCallback(async (): Promise<boolean> => {
    if (!isOnboardingEnabled()) {
      return false
    }

    const previousNewChatProfile = $newChatProfile.get()
    const previousNewChatRoute = $newChatRoute.get()
    const previousProfile = $activeGatewayProfile.get()
    const previousConnectionId = activeGatewayConnectionId()
    const previousSetupSession = $setupSession.get()
    const previousThreadIds = $chatOnboardingThreadIds.get()
    let swapped = false

    try {
      await ensureSetupProfile(requestGateway)

      // Probe the guide's own socket before switching profiles so a refusal
      // leaves classic onboarding on the user's current backend.
      const record = await requestGatewayForProfile<SetupStatus>(SETUP_PROFILE, 'setup.status', {})

      if (record.ready !== true || record.provider_configured !== true) {
        return false
      }

      swapped = true
      $newChatRoute.set(null)
      $newChatProfile.set(SETUP_PROFILE)
      await ensureGatewayProfile(SETUP_PROFILE)

      startChatOnboardingSolo()
      window.hermesDesktop?.chatOnboarding?.soloBoot?.()
      await loadMachineProfile()

      const seedMessages = buildChatOnboardingSeedMessages(
        pickOnboardingGreeting(),
        record.provider_configured === true && record.free_tier !== true
      )

      const guideRequest: OnboardingHandoffOptions['requestGateway'] = (method, params, timeout) =>
        requestGatewayForProfile(SETUP_PROFILE, method, params, timeout)

      // The exact title is the durable registry: a relaunch adopts the guide
      // before creating, so UNIQUE(title) cannot strand an untitled duplicate.
      const registryHit = await guideRequest<{ sessions?: GuideSession[] }>('session.list', {
        include_hidden: true,
        title: SETUP_CHAT_TITLE
      })

      const canonical = registryHit?.sessions?.[0]

      if (canonical?.id) {
        await adoptGuideSession(canonical, record.free_tier, resumeSession, guideRequest)

        // runGuideKickoff records the guided phase only after adoption.
        return true
      }

      const createOverrides: SessionCreateOverrides = { title: SETUP_CHAT_TITLE }

      if (record.free_tier) {
        createOverrides.reasoningEffort = 'minimal'
      }

      const runtimeId = await runCreatePinnedTo(SETUP_PROFILE, () =>
        createBackendSessionForSend(null, seedMessages, createOverrides)
      )

      if (!runtimeId) {
        throw new Error('The welcome chat could not be created. Please try again.')
      }

      const storedId = $selectedStoredSessionId.get()
      $chatOnboardingThreadIds.set(storedId ? [storedId, runtimeId] : [runtimeId])
      $setupSession.set({
        connectionId: guideSourceConnectionId(storedId),
        profile: SETUP_PROFILE,
        runtimeId,
        storedId
      })

      // Manual title authority prevents the hidden runbook becoming the title.
      await guideRequest('session.title', { session_id: runtimeId, title: SETUP_CHAT_TITLE }).catch(() => undefined)

      // session.create persisted both seed rows before the phase can advance.
      return true
    } catch (error) {
      $newChatProfile.set(previousNewChatProfile)
      $newChatRoute.set(previousNewChatRoute)
      $setupSession.set(previousSetupSession)
      $chatOnboardingThreadIds.set(previousThreadIds)
      endChatOnboardingSolo()
      skipGuide()

      if (swapped) {
        await (
          previousConnectionId
            ? ensureGatewayAgent(previousConnectionId, previousProfile)
            : ensureGatewayProfile(previousProfile)
        ).catch(restoreError => {
          notify({ kind: 'error', title: 'Could not restore your profile', message: String(restoreError) })
        })
      }

      console.error('[setup] welcome chat could not start', error)
      notify({
        kind: 'error',
        title: 'Welcome chat needs attention',
        message: error instanceof Error ? error.message : 'The welcome chat could not start.'
      })

      return false
    }
  }, [createBackendSessionForSend, requestGateway, resumeSession, runCreatePinnedTo])

  // The receipt survives failure and relaunch; only a confirmed go signal
  // completes onboarding. Never fall back to building in the guide chat.
  const setupHandoff = useStore($setupHandoff)
  const selectedStoredId = useStore($selectedStoredSessionId)

  // Resume only an EXISTING receipt when the welcome chat is reopened after
  // relaunch. Replayed directives stay inert; recovery never mints a new build.
  useEffect(() => {
    if (
      !isOnboardingEnabled() ||
      $setupHandoff.get() ||
      !selectedStoredId ||
      $activeGatewayProfile.get() !== SETUP_PROFILE
    ) {
      return
    }

    const connectionId = guideSourceConnectionId(selectedStoredId)

    try {
      const saved = readHandoffReceipt(handoffReceiptKey(connectionId, selectedStoredId))

      if (!saved) {
        return
      }

      if (saved.status === 'accepted') {
        completeOnboardingFlow()
        $setupHandoff.set({
          task: saved.task,
          brief: saved.brief,
          plan: saved.plan,
          phase: 'done',
          sessionTitle: firstTaskTitle(saved.task)
        })

        return
      }

      $setupSession.set({
        connectionId,
        profile: SETUP_PROFILE,
        runtimeId: $activeSessionId.get() ?? '',
        storedId: selectedStoredId
      })
      $setupHandoff.set({ task: saved.task, brief: saved.brief, plan: saved.plan, phase: 'pending' })
    } catch (error) {
      notify({
        kind: 'error',
        title: 'First build needs attention',
        message: error instanceof Error ? error.message : 'The first-build receipt could not be read.'
      })
    }
  }, [selectedStoredId])

  // Rebind the runtime pointer after session.resume; this is not an atom-to-ref mirror.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (!isOnboardingEnabled() || setupHandoff?.phase !== 'pending' || $setupHandoff.get() !== setupHandoff) {
      return
    }

    beginOnboardingHandoff()
    $setupHandoff.set({ ...setupHandoff, phase: 'opening' })

    void (async () => {
      const setupSession = setupHandoff.guide ?? $setupSession.get()
      const connectionId = setupSession?.connectionId ?? null

      const signpost = !declinedLookAround($messages.get())
      const previousNewChatProfile = $newChatProfile.get()
      const previousNewChatRoute = $newChatRoute.get()
      let receipt: HandoffReceipt | null = null
      let receiptKey: string | null = null

      const request: HandoffDeps['request'] = (owner, method, params) =>
        requestGatewayForAgent(owner.connectionId, owner.profile, method, params, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)

      try {
        if (!setupSession?.storedId) {
          throw new Error('The welcome chat owner is not available yet. Reopen it and retry the first build.')
        }

        $setupSession.set(setupSession)
        receiptKey = handoffReceiptKey(connectionId, setupSession.storedId)
        receipt = readHandoffReceipt(receiptKey)
        const owner: HandoffReceipt['owner'] = receipt?.owner ?? { connectionId, profile: BUILD_PROFILE }
        // Save facts before session.create freezes the new agent's memory.
        // A retry never re-creates the session or copies the guide's memory.
        receipt = await startHandoff(
          {
            read: () => receipt,
            save: value => {
              receipt = value
              saveHandoffReceipt(receiptKey!, value)
            },
            personalize: async () => {
              const result = await request<{ saved?: boolean; profile?: string; target?: string }>(
                owner,
                'profiles.remember_onboarding',
                { answers: $onboardingAnswers.get() }
              )

              if (!result.saved || result.profile !== BUILD_PROFILE || result.target !== 'user') {
                throw new Error('Could not save your onboarding preferences. Retry before starting the first build.')
              }
            },
            create: async () => {
              await ensureGatewayAgent(owner.connectionId, owner.profile)
              $newChatProfile.set(BUILD_PROFILE)
              // An ambient owner pins no route on the new chat either; the
              // profile is the whole address.
              $newChatRoute.set(
                owner.connectionId ? { connectionId: owner.connectionId, profile: owner.profile } : null
              )

              const seed = await buildFirstTaskSeedMessages(
                setupHandoff.task,
                $onboardingAnswers.get(),
                setupHandoff.plan
              )

              const runtimeId = await runCreatePinnedTo(BUILD_PROFILE, () =>
                createBackendSessionForSend(setupHandoff.brief, seed)
              )

              if (!runtimeId) {
                throw new Error('Could not open the first-build session.')
              }

              // Selection is usable only while it still names THIS create, not
              // whichever thread the user clicked while the request was away.
              const storedId =
                ensureSessionState(runtimeId).storedSessionId ??
                ($activeSessionId.get() === runtimeId ? $selectedStoredSessionId.get() : null)

              if (!storedId || storedId === setupSession.storedId) {
                throw new Error(
                  'The first-build session did not return a durable identity. Check your sessions before retrying.'
                )
              }

              return { runtimeId, storedId, owner }
            },
            request,
            bind: (value, running, snapshot) => {
              // An ambient owner (null connection id) has no route to pin: the
              // build session is reached the way any session on that profile
              // is, through the profile resolver.
              if (value.owner.connectionId) {
                const ownerRoute = { connectionId: value.owner.connectionId, profile: value.owner.profile }

                setSessionOwnerHint(value.storedId, ownerRoute)
                patchSessionTile(value.storedId, { runtimeId: value.runtimeId, ownerRoute })
              } else {
                // Explicit on both records: the tile patch merges, so an
                // omitted route would keep whatever a previous bind left there.
                forgetSessionOwnerHintsForSession(value.storedId)
                patchSessionTile(value.storedId, { runtimeId: value.runtimeId, ownerRoute: undefined })
              }

              ensureSessionState(value.runtimeId, value.storedId)
              updateSessionState(
                value.runtimeId,
                state =>
                  snapshot
                    ? {
                        ...state,
                        messages: toChatMessages(snapshot.messages ?? []),
                        busy: running,
                        awaitingResponse: running
                      }
                    : paintHandoffBrief(state, value.brief, value.storedId),
                value.storedId
              )

              // Rebind a reclaimed runtime only if the user still has this
              // stored chat selected. Background recovery must not steal focus.
              if ($selectedStoredSessionId.get() === value.storedId) {
                activeSessionIdRef.current = value.runtimeId
                setActiveSessionId(value.runtimeId)
                setAwaitingResponse(running)
                setBusy(running)
              }
            }
          },
          setupHandoff
        )

        // Naming is not identity and must not gate submission. Create's title
        // is pending metadata on older backends, so explicitly title afterward.
        const chatTitle = firstTaskTitle(receipt.task)
        await request(receipt.owner, 'session.title', { session_id: receipt.runtimeId, title: chatTitle }).catch(
          error => console.warn('[handoff] title could not be saved', error)
        )
        markSetupHandoffDone()
        completeOnboardingFlow()
        $handoffError.set(null)
        dismissNotification('onboarding-handoff')
        $setupHandoff.set({
          brief: receipt.brief,
          phase: 'done',
          plan: receipt.plan,
          sessionTitle: chatTitle,
          task: receipt.task
        })
        watchFirstBuild(receipt.runtimeId, receipt.owner.profile)
        const tree = $layoutTree.get()
        const sessionsGroup = tree ? findGroupOfPane(tree, 'sessions') : null

        if (sessionsGroup && sessionsGroup.active !== 'sessions') {
          activateTreePane(sessionsGroup.id, 'sessions')
        }

        // This is an informational success note, never an alternate build.
        void requestGatewayForAgent(
          connectionId,
          setupSession.profile ?? BUILD_PROFILE,
          'prompt.submit',
          {
            display_kind: 'hidden',
            session_id: setupSession.runtimeId,
            text: buildHandoffCompleteNote(receipt.task)
          },
          PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
        ).catch(error => console.warn('[handoff] guide note was not delivered', error))

        if (signpost && $selectedStoredSessionId.get() === receipt.storedId) {
          void showProfileSignpost()
        }
      } catch (error) {
        console.error('[handoff] first build needs recovery', error)

        if (receipt) {
          const briefId = `user-handoff-brief-${receipt.storedId}`
          updateSessionState(
            receipt.runtimeId,
            state => ({
              ...state,
              busy: false,
              awaitingResponse: false,
              turnStartedAt: null,
              messages: state.messages.filter(message => message.id !== briefId)
            }),
            receipt.storedId
          )

          if ($selectedStoredSessionId.get() === receipt.storedId) {
            setAwaitingResponse(false)
            setBusy(false)
          }
        }

        $newChatProfile.set(previousNewChatProfile)
        $newChatRoute.set(previousNewChatRoute)
        const message = error instanceof Error ? error.message : 'The first build could not be started.'
        $handoffError.set(message)
        $setupHandoff.set({ ...setupHandoff, phase: 'error' })
        notify({
          id: 'onboarding-handoff',
          kind: 'error',
          title: 'First build needs attention',
          message,
          action: { label: 'Retry first build', onClick: retrySetupHandoff }
        })
      }
    })()
  }, [
    activeSessionIdRef,
    createBackendSessionForSend,
    ensureSessionState,
    runCreatePinnedTo,
    setupHandoff,
    updateSessionState
  ])

  // The during-task check-in: first-build.ts decides WHEN (see its header),
  // this delivers it — a hidden note into the build's own session, which the
  // agent answers as a short status plus one ask.
  const checkIn = useStore($setupCheckIn)

  useEffect(() => {
    if (!isOnboardingEnabled() || !checkIn) {
      return
    }

    // The session dispatcher resolves the stored owner hint and runtime map
    // published by the handoff, including its exact registry connection.
    void requestGateway(
      'prompt.submit',
      { display_kind: 'hidden', session_id: checkIn.sessionId, text: checkIn.note },
      PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
    ).catch(() => undefined)
  }, [checkIn, requestGateway])

  return kickoffFirstChat
}
