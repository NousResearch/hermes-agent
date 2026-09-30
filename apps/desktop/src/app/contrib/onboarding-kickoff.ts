import type { OnboardingEnsureSetupProfileResult, OnboardingEnsureSetupSessionResult } from '@hermes/shared'
import { useCallback } from 'react'

import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import { $chatOnboardingThreadIds, endChatOnboardingSolo } from '@/components/onboarding-chat/assembly'
import { $introTurnSent, openIntro, rememberLaunchSource } from '@/components/onboarding-chat/intro'
import { $setupSession, guideSourceConnectionId } from '@/components/onboarding-chat/setup-profile'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { prefetchConnectorCatalog } from '@/store/connector-catalog'
import { activeGatewayConnectionId, requestGatewayForProfile } from '@/store/gateway'
import { notify } from '@/store/notifications'
import { $setupProfileName, type GuideKickoffResult } from '@/store/onboarding-gate'
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

import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

function prefetchGuideCatalogs(storedId: null | string): void {
  if (storedId) {
    prefetchConnectorCatalog(storedId)
    prefetchOnboardingPlugins(storedId)
  }
}

/** Runs a slash command as if typed; `hidden` keeps its user row out of every transcript. */
export type KickoffSlashCommand = (command: string, options: { hidden: true; sessionId: string }) => Promise<void>

export interface OnboardingKickoffOptions extends Pick<ReturnType<typeof useSessionActions>, 'resumeSession'> {
  requestGateway: AmbientGatewayRequest
  runSlashCommand: KickoffSlashCommand
}

interface SetupStatus {
  ready?: boolean
  provider_configured?: boolean
  free_tier_route?: boolean
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
    let swapped = false

    try {
      const { name: setupProfile } = await requestGateway<OnboardingEnsureSetupProfileResult>(
        'onboarding.ensure_setup_profile',
        {}
      )

      $setupProfileName.set(setupProfile)

      const record = await requestGatewayForProfile<SetupStatus>(setupProfile, 'setup.status', {})

      if (record.ready !== true || record.provider_configured !== true) {
        return 'off'
      }

      rememberLaunchSource()
      swapped = true
      $newChatRoute.set(null)
      $newChatProfile.set(setupProfile)
      await ensureGatewayProfile(setupProfile)

      const guideRequest: AmbientGatewayRequest = (method, params, timeout) =>
        requestGatewayForProfile(setupProfile, method, params, timeout)

      // Finds the setup chat by title or creates it; `empty` is true only before its first turn.
      const setupChat = await requestGateway<OnboardingEnsureSetupSessionResult>('onboarding.ensure_setup_session', {})

      const runtimeId = await adoptGuideSession(
        setupProfile,
        setupChat.session_id,
        record.free_tier_route,
        resumeSession,
        guideRequest
      )

      openIntro(setupChat.empty)

      if (setupChat.empty) {
        // The skill turn: the backend plays the fixed cards, then the model takes over.
        void runSlashCommand('/initiate-setup', { hidden: true, sessionId: runtimeId }).finally(() =>
          $introTurnSent.set(true)
        )
      }

      return 'started'
    } catch (error) {
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

      console.error('[setup] welcome chat could not start', error)
      notify({
        kind: 'error',
        title: 'Welcome chat needs attention',
        message: error instanceof Error ? error.message : 'The welcome chat could not start.'
      })

      return 'failed'
    }
  }, [requestGateway, resumeSession, runSlashCommand])
}
