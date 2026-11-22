import './discovery-lab.css'

import { useStore } from '@nanostores/react'
import { computed } from 'nanostores'
import { Navigate, useSearchParams } from 'react-router'

import { Button } from '@/components/ui/button'
import { $artifactRegistry } from '@/store/artifacts'
import { discoveryLabSignUpOn } from '@/store/discovery-lab'
import { $interfaceMode } from '@/store/interface-mode'
import { $sessions } from '@/store/session'
import { usePortalShell } from '@/store/shell-variant'

import { PageHeader } from '../redesign/page-header'
import { NEW_CHAT_ROUTE } from '../routes'
import { ListRow } from '../settings/primitives'
import { SettingsCard, SettingsGroup, SettingsOverview } from '../settings/settings-card'

import { LAB_TITLE } from './lab-copy'
import { useLabHost } from './lab-host'
import {
  endSignUpStandIn,
  LAB_GROUPS,
  LAB_STATES,
  type LabFacts,
  type LabHost,
  type LabState,
  resetPerson
} from './lab-states'
import { LabReadout } from './readout'
import { SignUpStandIn } from './sign-up-stand-in'

/** The newest chat the sidebar lists (a routine's own runs and hidden or archived chats are not one). */
const $newestChatId = computed($sessions, sessions => {
  const listed = sessions.filter(row => !row.hidden && !row.archived && !row.scheduler_owned && !row.cron_execution_id)

  return listed.reduce<(typeof listed)[number] | null>(
    (newest, row) => (!newest || row.last_active > newest.last_active ? row : newest),
    null
  )?.id ?? null
})

/** A chat this window has seen an answer with code in (its code card registered it). */
const $codeChatId = computed(
  $artifactRegistry,
  registry => Object.entries(registry).find(([, records]) => records.some(record => record.kind === 'code'))?.[0] ?? null
)

function useLabFacts(): LabFacts {
  return {
    newestChatId: useStore($newestChatId),
    codeChatId: useStore($codeChatId),
    simpleMode: useStore($interfaceMode) === 'simple',
    signUpOn: discoveryLabSignUpOn()
  }
}

interface LabRowProps {
  state: LabState
  facts: LabFacts
  host: LabHost
}

function LabRow({ facts, host, state }: LabRowProps) {
  const unavailable = state.unavailable?.(facts) ?? null

  return (
    <ListRow
      action={
        <Button
          aria-label={`Show ${state.title}`}
          disabled={unavailable !== null}
          onClick={() => state.open(host, facts)}
          variant="outline"
        >
          Show
        </Button>
      }
      below={
        <p className="discovery-lab-note">
          {unavailable ?? (state.stores ? `Changes: ${state.stores}.` : 'Changes nothing stored: gone on reload.')}
        </p>
      }
      description={state.description}
      title={state.title}
    />
  )
}

/** Surfaces the lab cannot open honestly, and why. */
const LEFT_OUT = [
  {
    title: 'App update: details, progress and its tray row',
    why: 'Their only action installs the update for real. Without a real update waiting, a stand-in would be a dead Update now or a real install.'
  },
  {
    title: '"Good to see you again." (back after a month)',
    why: 'It follows the newest chat\'s last activity on the machine; the lab does not age chats.'
  }
]

function DiscoveryLabPage() {
  const facts = useLabFacts()
  const host = useLabHost()

  return (
    <section aria-label={LAB_TITLE} className="discovery-lab-page" data-page-surface="">
      <div className="discovery-lab-stage page-stage" data-scrolls="">
        <PageHeader
          description="Testing only. Puts this window into each onboarding and discovery state and takes you to where it shows. Everything it opens is the real UI."
          title={LAB_TITLE}
        />
        <SettingsOverview>
          <SettingsGroup
            description="The guidance marks are stored for this Hermes instance, so they are shared by every window and browser that opens it. Resetting them here resets them there too."
            title="Start over"
          >
            <SettingsCard>
              <ListRow
                action={
                  <Button onClick={() => resetPerson(host, 'new')} variant="outline">
                    Reset
                  </Button>
                }
                below={<p className="discovery-lab-note">Changes: every guidance mark and the held message.</p>}
                description="Clears every mark and replays a first visit to a new install: the welcome splash opens on a new chat."
                title="Reset to a brand-new person"
              />
              <ListRow
                action={
                  <Button onClick={() => resetPerson(host, 'returning')} variant="outline">
                    Reset
                  </Button>
                }
                below={<p className="discovery-lab-note">Changes: every guidance mark and the held message.</p>}
                description="Clears every mark and replays a first visit after the redesign by someone with chats: the rename splash opens."
                title="Reset to a returning user"
              />
              {facts.signUpOn && (
                <ListRow
                  action={
                    <Button onClick={() => endSignUpStandIn(host)} variant="outline">
                      Turn off
                    </Button>
                  }
                  below={<p className="discovery-lab-note">Reloads the page.</p>}
                  description="Sign up is required in this tab because of the lab. Turn it off to be signed in again."
                  title="Sign-up stand-in is on"
                />
              )}
            </SettingsCard>
          </SettingsGroup>

          <LabReadout signUpOn={facts.signUpOn} />

          {LAB_GROUPS.map(group => (
            <SettingsGroup description={group.description} key={group.id} title={group.title}>
              <SettingsCard>
                {LAB_STATES.filter(state => state.group === group.id).map(state => (
                  <LabRow facts={facts} host={host} key={state.id} state={state} />
                ))}
              </SettingsCard>
            </SettingsGroup>
          ))}

          <SettingsGroup description="Reachable only with backend support the lab cannot fake cleanly." title="Not in the lab">
            <SettingsCard>
              {LEFT_OUT.map(item => (
                <ListRow description={item.why} key={item.title} title={item.title} />
              ))}
            </SettingsCard>
          </SettingsGroup>
        </SettingsOverview>
      </div>
    </section>
  )
}

/** `#/discovery-lab` (TESTING ONLY): every onboarding and discovery state, one button each. With the
 *  sign-up stand-in on, `?stand-in=sign-up` is the stand-in sign-up page. The redesign's onboarding
 *  only, so classic goes to a new chat. */
export function DiscoveryLabView() {
  const portal = usePortalShell()
  const [params] = useSearchParams()

  if (!portal) {
    return <Navigate replace to={NEW_CHAT_ROUTE} />
  }

  return params.get('stand-in') === 'sign-up' && discoveryLabSignUpOn() ? <SignUpStandIn /> : <DiscoveryLabPage />
}
