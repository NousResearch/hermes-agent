import { Button } from '@/components/ui/button'
import { SIGN_IN_OUTCOME_PARAM } from '@/lib/visitor-access'
import { setDiscoveryLabSignUp, SIGN_UP_RETURN_PARAM } from '@/store/discovery-lab'

import { PageHeader } from '../redesign/page-header'
import { NEW_CHAT_ROUTE } from '../routes'
import { ListRow } from '../settings/primitives'
import { SettingsCard, SettingsOverview } from '../settings/settings-card'

import { useLabHost } from './lab-host'
import { type LabHost, loadAt } from './lab-states'

/** Where sign up was started from (`return_to`), only ever this app: anything else is a new chat. */
function returnTo(): URL {
  const here = new URL(window.location.href)
  const fallback = new URL(here.href)
  fallback.search = ''
  fallback.hash = NEW_CHAT_ROUTE

  try {
    const back = new URL(here.searchParams.get(SIGN_UP_RETURN_PARAM) ?? '', here)

    return back.origin === here.origin && back.pathname === here.pathname ? back : fallback
  } catch {
    return fallback
  }
}

/** Come back signed in: no sign up is required on the next load, as after a real sign up. */
function finishSignUp(host: LabHost): void {
  setDiscoveryLabSignUp(false)
  loadAt(host, returnTo().href)
}

/** Come back the way sign up does when it could not sign the visitor in. */
function comeBackSignedOut(host: LabHost): void {
  const back = returnTo()
  back.searchParams.set(SIGN_IN_OUTCOME_PARAM, 'failed')
  loadAt(host, back.href)
}

/**
 * TESTING ONLY: the lab's stand-in for the sign-up service, where every Sign up in the app leads while
 * the stand-in is on. It sends the visitor back to where they started either way, as the real one
 * does (lib/visitor-access.ts), and nothing leaves the app.
 */
export function SignUpStandIn() {
  const host = useLabHost()

  return (
    <section aria-label="Sign up stand-in" className="discovery-lab-page" data-page-surface="">
      <div className="discovery-lab-stage page-stage" data-scrolls="">
        <PageHeader
          description="Testing only. In the hosted Webapp, the sign-up service takes over here. This page stands in for it: choose how sign up ends."
          title="Sign up (stand-in)"
        />
        <SettingsOverview>
          <SettingsCard>
            <ListRow
              action={<Button onClick={() => finishSignUp(host)}>Finish sign up</Button>}
              description="Come back signed in. Sign up is no longer required, and a held message runs first, as a new chat."
              title="Sign up succeeds"
            />
            <ListRow
              action={
                <Button onClick={() => comeBackSignedOut(host)} variant="outline">
                  Come back
                </Button>
              }
              description={'Come back without being signed in: "Let\'s try signing in again", with the held message still held.'}
              title="Sign up comes back without signing in"
            />
          </SettingsCard>
        </SettingsOverview>
      </div>
    </section>
  )
}
