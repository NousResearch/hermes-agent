import { useStore } from '@nanostores/react'

import { $discoveryLabUsageStatsConsent } from '@/store/discovery-lab'
import { $guidanceLedger, FINISHED_TASK_CHATS_KEPT, type GuidanceTopic } from '@/store/guidance-ledger'
import { $heldPrompt } from '@/store/held-prompt'
import { $newVersionReady } from '@/store/new-version'

import { SettingsCard, SettingsGroup } from '../settings/settings-card'

const TOPICS: readonly { topic: GuidanceTopic; label: string }[] = [
  { topic: 'renameSplash', label: 'Rename splash' },
  { topic: 'firstTimeSplash', label: 'Welcome splash' },
  { topic: 'starterIdeas', label: 'Starter ideas' },
  { topic: 'usageStatsOffer', label: 'Usage-stats card' },
  { topic: 'checkIn', label: 'Check-in' },
  { topic: 'developerViewOffer', label: 'Developer-view offer' }
]

const NO_MARK = 'no mark'

interface ReadoutRowProps {
  label: string
  value: string
}

function ReadoutRow({ label, value }: ReadoutRowProps) {
  return (
    <div className="discovery-lab-readout-row">
      <dt>{label}</dt>
      <dd data-empty={value === NO_MARK || value === 'off' || value === 'none' || undefined}>{value}</dd>
    </div>
  )
}

/** What the lab can change, as it stands now: the stored guidance marks, then the stand-ins and the
 *  held message. */
export function LabReadout({ signUpOn }: { signUpOn: boolean }) {
  const ledger = useStore($guidanceLedger)
  const usageStandIn = useStore($discoveryLabUsageStatsConsent) !== null
  const newerBuild = useStore($newVersionReady)
  const held = useStore($heldPrompt)

  return (
    <SettingsGroup description="Updates as you use the app." title="Current marks">
      <SettingsCard>
        <dl aria-label="Guidance marks" className="discovery-lab-readout">
          {TOPICS.map(({ label, topic }) => (
            <ReadoutRow key={topic} label={label} value={ledger[topic] ?? NO_MARK} />
          ))}
          <ReadoutRow
            label="Chats with a finished task"
            value={`${ledger.finishedTaskChats?.length ?? 0} of ${FINISHED_TASK_CHATS_KEPT}`}
          />
        </dl>
        <dl aria-label="Stand-ins" className="discovery-lab-readout">
          <ReadoutRow label="Sign-up stand-in (this tab)" value={signUpOn ? 'on' : 'off'} />
          <ReadoutRow label="Usage-stats answer" value={usageStandIn ? 'stand-in: not answered' : 'the agent\'s own'} />
          <ReadoutRow label="Newer build waiting" value={newerBuild ? 'yes' : 'none'} />
          <ReadoutRow label="Held message (this browser)" value={held ?? 'none'} />
        </dl>
      </SettingsCard>
    </SettingsGroup>
  )
}
