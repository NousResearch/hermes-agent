import { getGlobalModelInfo, getGlobalModelOptions, getProfiles } from '@/hermes'
import type { ProfileScope } from '@/hermes'
import { setMainModelAssignment } from '@/store/cron-model-impact'
import { readableError } from '@/store/notifications'

export interface ProfileModelResult {
  profile: string
  ok: boolean
  error?: string
}

/** A one-time main-model assignment, never profile inheritance or a config copy. */
export async function applyModelToProfiles({
  connectionId,
  provider,
  model,
  isCurrent,
  onResult,
  unavailable,
  interrupted,
  failed
}: {
  connectionId: string
  provider: string
  model: string
  isCurrent: () => boolean
  onResult: (result: ProfileModelResult) => void
  unavailable: string
  interrupted: string
  failed: string
}): Promise<ProfileModelResult[]> {
  const assertCurrent = () => {
    if (!isCurrent()) {
      throw new Error(interrupted)
    }
  }

  assertCurrent()
  const { profiles } = await getProfiles({ connectionId })
  const results: ProfileModelResult[] = []

  for (const profile of profiles) {
    let result: ProfileModelResult

    try {
      assertCurrent()
      const scope: ProfileScope = { connectionId, profile: profile.name }
      const catalog = await getGlobalModelOptions(undefined, scope)
      assertCurrent()
      const target = catalog.providers?.find(row => row.slug === provider)

      // A matching slug can identify a DIFFERENT private endpoint in another
      // profile. Resolve that profile's endpoint; never copy keys or source URL.
      if (!target || target.authenticated === false || !target.models?.includes(model) || provider === 'moa') {
        throw new Error(unavailable)
      }

      const saved = await setMainModelAssignment(
        { provider, model, ...(target.api_url ? { base_url: target.api_url } : {}) },
        scope,
        { skipConfirmPrompt: true }
      )

      const verified = await getGlobalModelInfo(scope)

      if (verified.provider !== (saved.provider || provider) || verified.model !== (saved.model || model)) {
        throw new Error(failed)
      }

      result = { profile: profile.name, ok: true }
    } catch (error) {
      result = { profile: profile.name, ok: false, error: readableError(error, failed).message }
    }

    results.push(result)
    onResult(result)
  }

  return results
}
