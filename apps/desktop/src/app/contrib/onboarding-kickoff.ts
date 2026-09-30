import type { OnboardingEnsureSetupProfileResult } from '@hermes/shared'
import { useCallback } from 'react'

import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import type { SessionCreateOverrides } from '@/app/session/hooks/use-session-actions/create-overrides'
import {
  $chatOnboardingThreadIds,
  endChatOnboardingSolo,
  pickOnboardingGreeting,
  takeGuideShape
} from '@/components/onboarding-chat/assembly'
import { $setupSession, guideSourceConnectionId, SETUP_CHAT_TITLE } from '@/components/onboarding-chat/setup-profile'
import { chatMessageText } from '@/lib/chat-messages'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { prefetchConnectorCatalog } from '@/store/connector-catalog'
import { activeGatewayConnectionId, requestGatewayForProfile } from '@/store/gateway'
import { loadMachineProfile } from '@/store/machine'
import { notify } from '@/store/notifications'
import { readOnboardingCapabilities } from '@/store/onboarding-capabilities'
import type { GuideKickoffResult } from '@/store/onboarding-gate'
import { prefetchOnboardingPlugins } from '@/store/onboarding-plugins'
import { buildChatOnboardingSeedMessages } from '@/store/onboarding-script'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  ensureGatewayAgent,
  ensureGatewayProfile
} from '@/store/profile'
import { $activeSessionId, $messages, $selectedStoredSessionId } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

function prefetchGuideCatalogs(storedId: null | string, runtimeId: string): void {
  if (storedId) {
    prefetchConnectorCatalog(storedId, runtimeId)
    prefetchOnboardingPlugins(storedId)
  }
}

export interface OnboardingKickoffOptions extends Pick<
  ReturnType<typeof useSessionActions>,
  'createBackendSessionForSend' | 'resumeSession'
> {
  requestGateway: AmbientGatewayRequest
  runCreatePinnedTo: <T>(profile: string, create: () => Promise<T>) => Promise<T>
}

interface SetupStatus {
  ready?: boolean
  provider_configured?: boolean
  free_tier_route?: boolean
}

interface GuideSession {
  id: string
  resolved_id?: string
}

export async function adoptGuideSession(
  setupProfile: string,
  canonical: GuideSession,
  freeTierRoute: SetupStatus['free_tier_route'],
  resumeSession: OnboardingKickoffOptions['resumeSession'],
  guideRequest: AmbientGatewayRequest
): Promise<void> {
  await resumeSession(canonical.resolved_id ?? canonical.id, true)
  const adoptedRuntimeId = $activeSessionId.get()
  const state = adoptedRuntimeId ? $sessionStates.get()[adoptedRuntimeId] : undefined

  if (
    !adoptedRuntimeId ||
    !state?.storedSessionId ||
    ![canonical.id, canonical.resolved_id].includes(state.storedSessionId) ||
    $selectedStoredSessionId.get() !== state.storedSessionId ||
    $activeGatewayProfile.get() !== setupProfile ||
    !$messages.get().some(message => message.role === 'assistant' && !message.hidden && chatMessageText(message).trim())
  ) {
    throw new Error('The welcome conversation could not be loaded. Please try again.')
  }

  $chatOnboardingThreadIds.set([canonical.id, adoptedRuntimeId])
  $setupSession.set({
    connectionId: guideSourceConnectionId(canonical.id),
    profile: setupProfile,
    runtimeId: adoptedRuntimeId,
    storedId: canonical.id
  })
  prefetchGuideCatalogs(canonical.id, adoptedRuntimeId ?? canonical.id)

  if (freeTierRoute) {
    await guideRequest('config.set', {
      session_id: adoptedRuntimeId,
      key: 'reasoning',
      value: 'minimal'
    })
  }
}

export function useOnboardingKickoff({
  createBackendSessionForSend,
  requestGateway,
  resumeSession,
  runCreatePinnedTo
}: OnboardingKickoffOptions) {
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

      const record = await requestGatewayForProfile<SetupStatus>(setupProfile, 'setup.status', {})

      if (record.ready !== true || record.provider_configured !== true) {
        return 'off'
      }

      swapped = true
      $newChatRoute.set(null)
      $newChatProfile.set(setupProfile)
      await ensureGatewayProfile(setupProfile)

      takeGuideShape()
      await loadMachineProfile()

      const guideRequest: AmbientGatewayRequest = (method, params, timeout) =>
        requestGatewayForProfile(setupProfile, method, params, timeout)

      const registryHit = await guideRequest<{ sessions?: GuideSession[] }>('session.list', {
        include_hidden: true,
        title: SETUP_CHAT_TITLE
      })

      const canonical = registryHit?.sessions?.[0]

      if (canonical?.id) {
        await adoptGuideSession(setupProfile, canonical, record.free_tier_route, resumeSession, guideRequest)

        return 'started'
      }

      const capabilities = await readOnboardingCapabilities({
        connectionId: previousConnectionId,
        profile: setupProfile
      })

      const seedMessages = buildChatOnboardingSeedMessages(
        pickOnboardingGreeting(),
        record.free_tier_route !== true,
        capabilities
      )

      const createOverrides: SessionCreateOverrides = { title: SETUP_CHAT_TITLE }

      if (record.free_tier_route) {
        createOverrides.reasoningEffort = 'minimal'
      }

      const runtimeId = await runCreatePinnedTo(setupProfile, () =>
        createBackendSessionForSend(null, seedMessages, createOverrides)
      )

      if (!runtimeId) {
        throw new Error('The welcome chat could not be created. Please try again.')
      }

      const storedId = $selectedStoredSessionId.get()
      $chatOnboardingThreadIds.set(storedId ? [storedId, runtimeId] : [runtimeId])
      prefetchGuideCatalogs(storedId, runtimeId)
      $setupSession.set({
        connectionId: guideSourceConnectionId(storedId),
        profile: setupProfile,
        runtimeId,
        storedId
      })

      await guideRequest('session.title', { session_id: runtimeId, title: SETUP_CHAT_TITLE }).catch(() => undefined)

      await adoptGuideSession(
        setupProfile,
        { id: storedId ?? runtimeId },
        record.free_tier_route,
        resumeSession,
        guideRequest
      )

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
  }, [createBackendSessionForSend, requestGateway, resumeSession, runCreatePinnedTo])
}
