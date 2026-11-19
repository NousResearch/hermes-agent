import { OrganizationSwitcher, SignIn, SignUp } from '@clerk/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { ErrorState } from '@/components/ui/error-state'
import { useI18n } from '@/i18n'
import type { WebappInstanceState } from '@/store/webapp-instance'

import { webappInstanceCopy } from './webapp-instance-copy'

interface WebappInstanceStatusProps {
  state: WebappInstanceState
  pending: boolean
  signedIn: boolean
  actionError: null | string
  onRetry: () => void
  onSignOut: () => void
}

function BrowserSignIn() {
  const [signUp, setSignUp] = useState(() => window.location.hash.includes('/sign-up'))

  useEffect(() => {
    const onHashChange = () => setSignUp(window.location.hash.includes('/sign-up'))
    window.addEventListener('hashchange', onHashChange)

    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  return signUp ? <SignUp routing="hash" signInUrl="#/sign-in" /> : <SignIn routing="hash" signUpUrl="#/sign-up" />
}

export function WebappInstanceStatus({
  state,
  pending,
  signedIn,
  actionError,
  onRetry,
  onSignOut
}: WebappInstanceStatusProps) {
  const { locale, t } = useI18n()
  const copy = webappInstanceCopy(locale)

  if (pending) {
    return <p role="status">{state.status === 'redirecting' ? copy.redirecting : copy.checking}</p>
  }

  if (!signedIn && state.status === 'signed-out') {
    return <BrowserSignIn />
  }

  const descriptions: Partial<Record<WebappInstanceState['status'], string>> = {
    missing: copy.missing,
    inactive: copy.inactive,
    unauthorized: copy.unauthorized,
    'organization-required': copy.organizationRequired
  }
  const description = descriptions[state.status] ?? state.error ?? actionError

  return (
    <div className="grid max-w-md gap-6">
      <div role="alert">
        <ErrorState description={description} title={copy.error} />
      </div>
      {signedIn && state.status !== 'unauthorized' && <OrganizationSwitcher />}
      <Button onClick={onRetry}>{t.common.retry}</Button>
      <Button asChild variant="outline">
        <a href="https://unitalk.ai">{copy.openUnitalk}</a>
      </Button>
      {signedIn && (
        <Button onClick={onSignOut} variant={state.status === 'unauthorized' ? 'outline' : 'ghost'}>
          {state.status === 'unauthorized' ? copy.signInAgain : copy.signOut}
        </Button>
      )}
    </div>
  )
}
