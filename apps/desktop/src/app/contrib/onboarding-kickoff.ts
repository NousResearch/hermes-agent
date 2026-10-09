import type { OnboardingEnsureSetupProfileResult, OnboardingEnsureSetupSessionResult } from '@hermes/shared'
import { useCallback } from 'react'

import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { $chatOnboardingThreadIds, endChatOnboardingSolo, takeGuideShape } from '@/components/onboarding-chat/assembly'
import { $introTurnSent, openIntro, rememberLaunchSource } from '@/components/onboarding-chat/intro'
import { $setupSession, guideSourceConnectionId } from '@/components/onboarding-chat/setup-session'
import type { ChatMessagePart } from '@/lib/chat-messages'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { BACKEND_BOOT_WAIT_TIMEOUT_MS } from '@/lib/with-timeout'
import { prefetchConnectorCatalog } from '@/store/connector-catalog'
import { activeGatewayConnectionId, requestGatewayForProfile } from '@/store/gateway'
import { notify } from '@/store/notifications'
import { $setupProfileName, type GuideKickoffResult } from '@/store/onboarding-gate'
import { $introStartExit } from '@/store/onboarding-intro'
import { prefetchOnboardingPlugins } from '@/store/onboarding-plugins'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  ensureGatewayAgent,
  ensureGatewayProfile
} from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { createKickoffMachine, KickoffFailure, KickoffSkipped, NO_DEADLINE } from './onboarding-kickoff-machine'
import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

function prefetchGuideCatalogs(storedId: null | string): void {
  if (storedId) {
    prefetchConnectorCatalog(storedId)
    prefetchOnboardingPlugins(storedId)
  }
}

/** Runs a slash command as if typed; `hidden` keeps its user row out of every transcript. */
export type KickoffSlashCommand = (command: string, options: { hidden: true; sessionId: string }) => Promise<void>

interface OnboardingKickoffOptions extends Pick<ReturnType<typeof useSessionActions>, 'resumeSession'> {
  requestGateway: AmbientGatewayRequest
  runSlashCommand: KickoffSlashCommand
}

interface SetupStatus {
  ready?: boolean
  provider_configured?: boolean
  free_tier_route?: boolean
}

const answeredSetupCard = (part: ChatMessagePart) =>
  part.type === 'tool-call' && part.toolName === 'setup_choose' && part.result !== undefined

/**
 * Where the setup chat's opening stands after a relaunch. `blank`: no assistant words yet. `stalled`: no
 * live turn, and nothing after the last answered card drives the chat on — the backend's name and accent
 * cards were not both answered, or the model never spoke after the last answer (the app quit mid-turn).
 */
function setupChatOpening(runtimeId: string): { blank: boolean; stalled: boolean } {
  const state = $sessionStates.get()[runtimeId]
  const parts = (state?.messages ?? []).filter(m => m.role === 'assistant' && !m.hidden).flatMap(m => m.parts)
  const blank = !parts.some(part => part.type === 'text' && part.text.trim())

  if (!state || state.busy || state.awaitingResponse || state.turnLive || state.needsInput) {
    return { blank, stalled: false }
  }

  const answers = parts.flatMap((part, index) => (answeredSetupCard(part) ? [index] : []))

  if (answers.length < 2) {
    return { blank, stalled: true }
  }

  const spoke = parts
    .slice(answers[answers.length - 1] + 1)
    .some(part => part.type === 'tool-call' || (part.type === 'text' && part.text.trim()))

  return { blank, stalled: !spoke }
}

export async function adoptGuideSession(
  setupProfile: string,
  storedId: string,
  freeTierRoute: SetupStatus['free_tier_route'],
  resumeSession: OnboardingKickoffOptions['resumeSession'],
  guideRequest: AmbientGatewayRequest
): Promise<string> {
  await resumeSession(storedId, true)
  const adoptedRuntimeId = $activeSessionId.get()
  const state = adoptedRuntimeId ? $sessionStates.get()[adoptedRuntimeId] : undefined

  if (
    !adoptedRuntimeId ||
    !state?.storedSessionId ||
    $selectedStoredSessionId.get() !== state.storedSessionId ||
    $activeGatewayProfile.get() !== setupProfile
  ) {
    throw new Error('The welcome conversation could not be loaded. Please try again.')
  }

  $chatOnboardingThreadIds.set([...new Set([storedId, state.storedSessionId, adoptedRuntimeId])])
  $setupSession.set({
    connectionId: guideSourceConnectionId(storedId),
    profile: setupProfile,
    runtimeId: adoptedRuntimeId,
    storedId
  })
  prefetchGuideCatalogs(storedId)

  if (freeTierRoute) {
    await guideRequest('config.set', {
      session_id: adoptedRuntimeId,
      key: 'reasoning',
      value: 'minimal'
    })
  }

  return adoptedRuntimeId
}

export function useOnboardingKickoff({ requestGateway, resumeSession, runSlashCommand }: OnboardingKickoffOptions) {
  return useCallback(async (): Promise<GuideKickoffResult> => {
    if (!isOnboardingEnabled()) {
      return 'off'
    }

    const previousNewChatProfile = $newChatProfile.get()
    const previousNewChatRoute = $newChatRoute.get()
    const previousProfile = $activeGatewayProfile.get()
    const previousConnectionId = activeGatewayConnectionId()
    const previousSetupSession = $setupSession.get()
    const previousThreadIds = $chatOnboardingThreadIds.get()
    const machine = createKickoffMachine()
    let swapped = false

    // Past the boot budget the start keeps waiting, and the starting screen offers a way out.
    const offerExit = window.setTimeout(
      () =>
        $introStartExit.set(() =>
          machine.stop(new KickoffSkipped('Setup was skipped while Hermes was still starting.'))
        ),
      BACKEND_BOOT_WAIT_TIMEOUT_MS
    )

    try {
      // Every call here is idempotent, so a retry repeats the whole step. The calls carry no deadline: a cold
      // first boot answers them late, and the machine fails on the signals that mean it never will.
      const prepared = await machine.attempt('preparing', async () => {
        const { name: setupProfile } = await requestGateway<OnboardingEnsureSetupProfileResult>(
          'onboarding.ensure_setup_profile',
          {},
          NO_DEADLINE,
          machine.signal
        )

        $setupProfileName.set(setupProfile)

        // Foreground: the user is watching the starting screen, so a cold spawn of the setup profile's
        // backend takes the reserved slot and skips the background redial cooldown.
        const record = await requestGatewayForProfile<SetupStatus>(
          setupProfile,
          'setup.status',
          {},
          NO_DEADLINE,
          machine.signal,
          { spawnPriority: 'foreground' }
        )

        if (record.ready !== true || record.provider_configured !== true) {
          // A retry after the swap saw a ready provider the first time; the swap has to be undone.
          if (swapped) {
            throw new KickoffFailure('Inference stopped reporting ready while the welcome chat opened.')
          }

          return null
        }

        if (!swapped) {
          takeGuideShape()
          rememberLaunchSource()
          swapped = true
          $newChatRoute.set(null)
          $newChatProfile.set(setupProfile)
        }

        await ensureGatewayProfile(setupProfile)

        // Finds the setup chat by title or creates it; `empty` is true only before its first turn.
        const setupChat = await requestGateway<OnboardingEnsureSetupSessionResult>(
          'onboarding.ensure_setup_session',
          {},
          NO_DEADLINE,
          machine.signal
        )

        return { record, setupChat, setupProfile }
      })

      if (!prepared) {
        machine.enter('off', 'no inference ready')

        return 'off'
      }

      const { record, setupChat, setupProfile } = prepared

      // The way out covers the waits only. Opening switches the session and cannot be cancelled halfway, so a
      // stop there would roll back under a resume that still lands on the setup chat.
      window.clearTimeout(offerExit)
      $introStartExit.set(null)

      machine.enter('opening', `session ${setupChat.session_id}`)

      const guideRequest: AmbientGatewayRequest = (method, params, timeout) =>
        requestGatewayForProfile(setupProfile, method, params, timeout)

      const runtimeId = await adoptGuideSession(
        setupProfile,
        setupChat.session_id,
        record.free_tier_route,
        resumeSession,
        guideRequest
      )

      // A relaunch reopens the same setup chat; one whose opening was cut off is sent the command again.
      const opening = setupChat.empty ? { blank: true, stalled: true } : setupChatOpening(runtimeId)

      openIntro(opening.blank)
      $introTurnSent.set(!opening.stalled)

      if (opening.stalled) {
        // The skill turn: the backend plays the fixed cards, then the model takes over.
        void runSlashCommand('/initiate-setup', { hidden: true, sessionId: runtimeId }).finally(() =>
          $introTurnSent.set(true)
        )
      }

      machine.enter('started')

      return 'started'
    } catch (error) {
      machine.enter('failed', error instanceof Error ? error.message : String(error))
      $newChatProfile.set(previousNewChatProfile)
      $newChatRoute.set(previousNewChatRoute)
      $setupSession.set(previousSetupSession)
      $chatOnboardingThreadIds.set(previousThreadIds)
      endChatOnboardingSolo()

      if (swapped) {
        await (
          previousConnectionId
            ? ensureGatewayAgent(previousConnectionId, previousProfile)
            : ensureGatewayProfile(previousProfile)
        ).catch(restoreError => {
          notify({ kind: 'error', title: 'Could not restore your profile', message: String(restoreError) })
        })
      }

      // Going on without setup is the person's choice, not an error to report.
      if (!(error instanceof KickoffSkipped)) {
        console.error('[setup] welcome chat could not start', error)
        notify({
          kind: 'error',
          title: 'Welcome chat needs attention',
          message: error instanceof Error ? error.message : 'The welcome chat could not start.'
        })
      }

      return 'failed'
    } finally {
      window.clearTimeout(offerExit)
      $introStartExit.set(null)
      machine.dispose()
    }
  }, [requestGateway, resumeSession, runSlashCommand])
}
