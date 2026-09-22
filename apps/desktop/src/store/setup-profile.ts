/**
 * The setup profile the guided onboarding runs in. The backend creates it and marks it with
 * `role: setup` (`onboarding.ensure_setup_profile`); the renderer never knows its name up front.
 * Kickoff records the name the backend returned; after a relaunch the roster's role answers.
 */

import type { OnboardingEnsureSetupProfileResult } from '@hermes/shared'
import { atom } from 'nanostores'

import type { AmbientGatewayRequest } from '@/app/contrib/session-rpc-dispatcher'
import { $profiles } from '@/store/profile'

export const $setupProfileName = atom<null | string>(null)

export async function ensureSetupProfile(request: AmbientGatewayRequest): Promise<string> {
  const { name } = await request<OnboardingEnsureSetupProfileResult>('onboarding.ensure_setup_profile', {})
  $setupProfileName.set(name)

  return name
}

export function setupProfileName(): null | string {
  return $setupProfileName.get() ?? $profiles.get().find(profile => profile.role === 'setup')?.name ?? null
}

export function isSetupProfile(name: null | string | undefined): boolean {
  return !!name && name === setupProfileName()
}

export function requireSetupProfileName(): string {
  const name = setupProfileName()

  if (!name) {
    throw new Error('The welcome chat has no setup profile yet.')
  }

  return name
}
