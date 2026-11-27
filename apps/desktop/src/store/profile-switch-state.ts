import { atom, batch } from 'nanostores'

// Rail profile switches (selectProfile / newSessionInProfile) as presentation
// state: which pick is in flight and which one ended in a failure the user can
// retry. Owned by those two actions; the chat's switch indicator, the rail
// squares and the composer's send-target hint read it. Behavioural swap state
// ($gatewaySwapTarget — transcript suppression, gone-session verdicts) stays in
// store/profile.

export interface ProfileSwitchFailure {
  profile: string
  /** Re-runs the activation that failed, without starting another fresh draft. */
  retry: () => void
}

export const $profileSwitchPending = atom<null | string>(null)
export const $profileSwitchFailure = atom<null | ProfileSwitchFailure>(null)

// Only the latest pick may settle the atoms: a superseded attempt resolving
// late must neither clear the newer spinner nor resurrect a failure.
let generation = 0

export function trackProfileSwitch<T>(profile: string, attempt: Promise<T>, retry: () => void): Promise<T> {
  const current = ++generation

  batch(() => {
    $profileSwitchFailure.set(null)
    $profileSwitchPending.set(profile)
  })

  return attempt.then(
    value => {
      if (current === generation) {
        $profileSwitchPending.set(null)
      }

      return value
    },
    (error: unknown) => {
      if (current === generation) {
        batch(() => {
          $profileSwitchPending.set(null)
          $profileSwitchFailure.set({
            profile,
            retry: () => {
              if (current === generation) {
                retry()
              }
            }
          })
        })
      }

      throw error
    }
  )
}

/** A new draft/source intent supersedes both settled failures and late attempts. */
export function invalidateProfileSwitch(): void {
  generation += 1
  batch(() => {
    $profileSwitchPending.set(null)
    $profileSwitchFailure.set(null)
  })
}

export function dismissProfileSwitchFailure(): void {
  $profileSwitchFailure.set(null)
}
