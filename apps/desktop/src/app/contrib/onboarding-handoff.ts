import { useStore } from '@nanostores/react'
import { useEffect } from 'react'

import { PROMPT_SUBMIT_REQUEST_TIMEOUT_MS } from '@/api/client'
import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { $setupCheckIn, watchFirstBuild } from '@/components/onboarding-chat/first-build'
import {
  $handoffError,
  $setupHandoff,
  $setupSession,
  buildFirstTaskSeedMessages,
  buildHandoffCompleteNote,
  firstTaskTitle,
  guideSourceConnectionId,
  readGuideHandoffReceipt,
  retrySetupHandoff
} from '@/components/onboarding-chat/setup-profile'
import { showHandoffTour } from '@/components/onboarding-chat/signpost'
import { findGroupOfPane } from '@/components/pane-shell/tree/model'
import { $layoutTree, activateTreePane } from '@/components/pane-shell/tree/store'
import { toChatMessages } from '@/lib/chat-messages'
import { connectorTitle } from '@/lib/connector-tools'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { requestGatewayForAgent } from '@/store/gateway'
import { dismissNotification, notify } from '@/store/notifications'
import { $onboardingAnswers } from '@/store/onboarding-answers'
import { beginOnboardingHandoff, completeOnboardingFlow } from '@/store/onboarding-gate'
import { watchPluginOutcomes } from '@/store/onboarding-plugin-outcomes'
import { $activeGatewayProfile, $newChatProfile, $newChatRoute, ensureGatewayAgent } from '@/store/profile'
import {
  $activeSessionId,
  $selectedStoredSessionId,
  forgetSessionOwnerHintsForSession,
  setActiveSessionId,
  setAwaitingResponse,
  setBusy,
  setSessionOwnerHint
} from '@/store/session'
import { $sessionStates, patchSessionTile } from '@/store/session-states'

import { BUILD_PROFILE, type HandoffDeps, type HandoffReceipt, paintHandoffBrief, startHandoff } from './handoff-leg'
import { saveHandoffReceipt } from './handoff-receipt'
import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

export interface OnboardingHandoffOptions extends Pick<
  Parameters<typeof useSessionActions>[0],
  'activeSessionIdRef' | 'ensureSessionState' | 'updateSessionState'
> {
  createBackendSessionForSend: ReturnType<typeof useSessionActions>['createBackendSessionForSend']
  requestGateway: AmbientGatewayRequest
  runCreatePinnedTo: <T>(profile: string, create: () => Promise<T>) => Promise<T>
}

export function useOnboardingHandoff({
  activeSessionIdRef,
  ensureSessionState,
  updateSessionState,
  createBackendSessionForSend,
  requestGateway,
  runCreatePinnedTo
}: OnboardingHandoffOptions) {
  const setupHandoff = useStore($setupHandoff)
  const selectedStoredId = useStore($selectedStoredSessionId)
  const activeSessionId = useStore($activeSessionId)

  useEffect(() => watchPluginOutcomes(() => $setupSession.get()?.runtimeId), [])

  useEffect(() => {
    if (
      !isOnboardingEnabled() ||
      $setupHandoff.get() ||
      !selectedStoredId ||
      !activeSessionId ||
      $sessionStates.get()[activeSessionId]?.storedSessionId !== selectedStoredId
    ) {
      return
    }

    const connectionId = guideSourceConnectionId(selectedStoredId)

    try {
      const { receipt: saved } = readGuideHandoffReceipt(selectedStoredId)

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
        profile: $activeGatewayProfile.get(),
        runtimeId: activeSessionId,
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
  }, [activeSessionId, selectedStoredId])

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

      const previousNewChatProfile = $newChatProfile.get()
      const previousNewChatRoute = $newChatRoute.get()
      let receipt: HandoffReceipt | null = null

      const request: HandoffDeps['request'] = (owner, method, params) =>
        requestGatewayForAgent(owner.connectionId, owner.profile, method, params, PROMPT_SUBMIT_REQUEST_TIMEOUT_MS)

      try {
        if (!setupSession?.storedId) {
          throw new Error('The welcome chat owner is not available yet. Reopen it and retry the first build.')
        }

        $setupSession.set(setupSession)
        const { key: receiptKey, receipt: saved } = readGuideHandoffReceipt(setupSession.storedId)
        receipt = saved
        const owner: HandoffReceipt['owner'] = receipt?.owner ?? { connectionId, profile: BUILD_PROFILE }
        receipt = await startHandoff(
          {
            read: () => receipt,
            save: value => {
              receipt = value
              saveHandoffReceipt(receiptKey, value)
            },
            personalize: async () => {
              const answers = $onboardingAnswers.get()

              const result = await request<{ saved?: boolean; profile?: string; target?: string }>(
                owner,
                'profiles.remember_onboarding',
                {
                  answers: { ...answers, connectors: answers.connectors.map(connectorTitle), plugins: answers.plugins }
                }
              )

              if (!result.saved || result.profile !== BUILD_PROFILE || result.target !== 'user') {
                throw new Error('Could not save your onboarding preferences. Retry before starting the first build.')
              }
            },
            create: async () => {
              await ensureGatewayAgent(owner.connectionId, owner.profile)
              $newChatProfile.set(BUILD_PROFILE)
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
              if (value.owner.connectionId) {
                const ownerRoute = { connectionId: value.owner.connectionId, profile: value.owner.profile }

                setSessionOwnerHint(value.storedId, ownerRoute)
                patchSessionTile(value.storedId, { runtimeId: value.runtimeId, ownerRoute })
              } else {
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

        const chatTitle = firstTaskTitle(receipt.task)
        await request(receipt.owner, 'session.title', { session_id: receipt.runtimeId, title: chatTitle }).catch(
          error => console.warn('[handoff] title could not be saved', error)
        )
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

        if ($selectedStoredSessionId.get() === receipt.storedId) {
          void showHandoffTour()
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

  const checkIn = useStore($setupCheckIn)

  useEffect(() => {
    if (!isOnboardingEnabled() || !checkIn) {
      return
    }

    void requestGateway(
      'prompt.submit',
      { display_kind: 'hidden', session_id: checkIn.sessionId, text: checkIn.note },
      PROMPT_SUBMIT_REQUEST_TIMEOUT_MS
    ).catch(() => undefined)
  }, [checkIn, requestGateway])
}
