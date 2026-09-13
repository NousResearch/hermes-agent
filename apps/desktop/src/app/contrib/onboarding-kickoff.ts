import { useCallback } from 'react'

import type { useSessionActions } from '@/app/session/hooks/use-session-actions'
import type { SessionCreateOverrides } from '@/app/session/hooks/use-session-actions/create-overrides'
import {
  $chatOnboardingThreadIds,
  endChatOnboardingSolo,
  pickOnboardingGreeting,
  startChatOnboardingSolo
} from '@/components/onboarding-chat/assembly'
import {
  $setupSession,
  ensureSetupProfile,
  guideSourceConnectionId,
  SETUP_CHAT_TITLE,
  SETUP_PROFILE
} from '@/components/onboarding-chat/setup-profile'
import { isOnboardingEnabled } from '@/lib/onboarding-enabled'
import { activeGatewayConnectionId, requestGatewayForProfile } from '@/store/gateway'
import { loadMachineProfile } from '@/store/machine'
import { notify } from '@/store/notifications'
import { skipGuide } from '@/store/onboarding-gate'
import { buildChatOnboardingSeedMessages } from '@/store/onboarding-script'
import {
  $activeGatewayProfile,
  $newChatProfile,
  $newChatRoute,
  ensureGatewayAgent,
  ensureGatewayProfile
} from '@/store/profile'
import { $activeSessionId, $selectedStoredSessionId } from '@/store/session'

import type { AmbientGatewayRequest } from './session-rpc-dispatcher'

export interface OnboardingKickoffOptions extends Pick<
  ReturnType<typeof useSessionActions>,
  'createBackendSessionForSend' | 'resumeSession'
> {
  requestGateway: AmbientGatewayRequest
  /** The caller's own requestGateway is what reads the pin. */
  runCreatePinnedTo: <T>(profile: string, create: () => Promise<T>) => Promise<T>
}

interface SetupStatus {
  ready?: boolean
  provider_configured?: boolean
  free_tier?: boolean
}

interface GuideSession {
  id: string
  resolved_id?: string
}

async function adoptGuideSession(
  canonical: GuideSession,
  freeTier: SetupStatus['free_tier'],
  resumeSession: OnboardingKickoffOptions['resumeSession'],
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

/** Seed the runbook and banked greeting on hermes-setup before advancing the phase.
 * The seeded assistant row opens the chat without a model turn. */
export function useOnboardingKickoff({
  createBackendSessionForSend,
  requestGateway,
  resumeSession,
  runCreatePinnedTo
}: OnboardingKickoffOptions) {
  return useCallback(async (): Promise<boolean> => {
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

      const seedMessages = buildChatOnboardingSeedMessages(pickOnboardingGreeting(), record.free_tier !== true)

      const guideRequest: AmbientGatewayRequest = (method, params, timeout) =>
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
}
