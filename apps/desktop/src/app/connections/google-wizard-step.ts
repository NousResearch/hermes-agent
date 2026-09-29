import type { GoogleStatus } from '@/api/google'

export type GoogleWizardStep = 'client' | 'consent' | 'verify'

/**
 * Where the wizard should be, from what is already in place — so reopening it
 * (or connecting after a half-finished attempt) resumes instead of starting over.
 */
export function googleWizardStep(status: Pick<GoogleStatus, 'client_secret' | 'token'> | null): GoogleWizardStep {
  if (!status?.client_secret) {
    return 'client'
  }

  return status.token ? 'verify' : 'consent'
}
