/** How a connected provider row should describe its credential.
 *
 *  The Accounts card and onboarding row share one flow subtitle ("sign in in
 *  your terminal") for every `external` provider. That is wrong once the
 *  backend reports an env-var API key: the user is not mid sign-in, and the
 *  credential is not owned by another CLI. */

export interface CredentialStatus {
  logged_in?: boolean
  source?: null | string
  source_label?: null | string
}

export function connectedCredentialLine(status: CredentialStatus | null | undefined, flowSubtitle: string): string {
  if (status?.logged_in && status.source === 'env_var' && status.source_label) {
    return status.source_label
  }

  return flowSubtitle
}

/** True when the remove hint should send the user to another program's CLI.
 *  An env-var credential is an API key Hermes itself stored. */
export function credentialRemoveIsCliManaged(status: CredentialStatus | null | undefined): boolean {
  return status?.source !== 'env_var'
}
