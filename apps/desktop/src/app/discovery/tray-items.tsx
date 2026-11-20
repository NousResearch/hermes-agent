import { computed } from 'nanostores'
import type { ComponentType } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Tip } from '@/components/ui/tooltip'
import { X } from '@/lib/icons'
import { $cloudOwnsUpdates, updateTargetFor } from '@/store/cloud-updates'
import { $firstTimeSplashReopenable, reopenFirstTimeSplash } from '@/store/first-time-splash'
import { $newVersionReady } from '@/store/new-version'
import { createNotificationTray, type TrayItemSource } from '@/store/notification-tray'
import { $renameSplashReopenable, reopenRenameSplash } from '@/store/rename-splash'
import { $connection } from '@/store/session'
import {
  $backendUpdateApply,
  $backendUpdateStatus,
  $updateApply,
  $updateStatus,
  startActiveUpdate
} from '@/store/updates'
import { $usageStatsOfferPlace, clearUsageStatsOffer, openUsageStatsDetails } from '@/store/usage-stats-offer'
import { $signUpRequired, startSignUp } from '@/store/visitor-access'

import { useDiscoveryCopy } from './discovery-copy'
import { useFirstTimeCopy } from './first-time-copy'
import { useUsageStatsCopy } from './usage-stats-copy'

export interface TrayCardProps {
  /** Closes the tray: call it before an action that takes the person somewhere else. */
  close: () => void
  /** Closes the tray and gives focus back to the bell: for an item that removes itself. */
  dismiss: () => void
  /** The item had news the person had not seen when the tray opened. */
  unseen: boolean
}

/** One entry of the notifications tray. Add an item by appending a definition to {@link TRAY_ITEMS};
 *  the tray, its bell and what counts as seen need no change. */
export interface TrayItemDefinition extends TrayItemSource {
  Card: ComponentType<TrayCardProps>
}

interface TrayCardViewProps {
  eyebrow?: string
  title: string
  description?: string
  action: { label: string; run: () => void }
  /** An × that takes the item out of the tray. */
  clear?: { label: string; run: () => void }
  unseen: boolean
}

/** The tray's one card shape: a caption, a display title, a sentence, one action. Items compose it
 *  so every entry reads alike; an item that needs a different shape renders its own. */
export function TrayCard({ action, clear, description, eyebrow, title, unseen }: TrayCardViewProps) {
  const copy = useDiscoveryCopy()

  return (
    <article className="discovery-tray-card" data-slot="tray-card" data-unseen={unseen || undefined}>
      {(eyebrow || unseen) && (
        <p className="discovery-tray-card-eyebrow">
          {unseen && <Badge variant="default">{copy.tray.newBadge}</Badge>}
          {eyebrow}
        </p>
      )}
      <h3 className="discovery-tray-card-title">{title}</h3>
      {description && <p className="discovery-tray-card-description">{description}</p>}
      <Button className="discovery-tray-card-action" onClick={action.run} size="sm" variant="outline">
        {action.label}
      </Button>
      {clear && (
        <Tip label={clear.label}>
          <Button
            aria-label={clear.label}
            className="discovery-tray-card-clear"
            onClick={clear.run}
            size="icon-xs"
            variant="ghost"
          >
            <X />
          </Button>
        </Tip>
      )}
    </article>
  )
}

/** An update for the target this window would install (the same choice as the palette's Update
 *  row), while it waits and nothing is installing it. Hermes Cloud updates its own agents. */
const $pendingUpdate = computed(
  [$connection, $cloudOwnsUpdates, $updateStatus, $backendUpdateStatus, $updateApply, $backendUpdateApply],
  (connection, cloudOwns, client, backend, clientApply, backendApply) => {
    const target = updateTargetFor(connection)
    const status = target === 'backend' ? backend : client
    const applying = (target === 'backend' ? backendApply : clientApply).applying

    if (cloudOwns[target] || applying || !status?.supported || status.error) {
      return null
    }

    if (!((status.behind ?? 0) > 0 || status.updateAvailable)) {
      return null
    }

    // What the update installs: a newer target is news again. Checks run at most daily in the
    // background (store/updates.ts), so a busy branch lights the bell no more often than that.
    return { news: status.targetSha ?? status.latestTag ?? 'available', target }
  }
)

function AppUpdateCard({ close, unseen }: TrayCardProps) {
  const copy = useDiscoveryCopy().updateItem

  return (
    <TrayCard
      action={{
        label: copy.action,
        run: () => {
          close()
          startActiveUpdate($pendingUpdate.get()?.target)
        }
      }}
      eyebrow={copy.eyebrow}
      title={copy.title}
      unseen={unseen}
    />
  )
}

function NewVersionCard({ unseen }: TrayCardProps) {
  const copy = useDiscoveryCopy().newVersion

  return (
    <TrayCard
      action={{ label: copy.refresh, run: () => window.location.reload() }}
      description={copy.description}
      eyebrow={copy.eyebrow}
      title={copy.title}
      unseen={unseen}
    />
  )
}

function RenameSplashCard({ close, unseen }: TrayCardProps) {
  const copy = useDiscoveryCopy().renameItem

  return (
    <TrayCard
      action={{
        label: copy.action,
        run: () => {
          close()
          reopenRenameSplash()
        }
      }}
      description={copy.description}
      title={copy.title}
      unseen={unseen}
    />
  )
}

/** Sign up (Onboarding 18695:19432), for as long as it is required: news until the tray has been opened.
 *  A sign up that came back without signing the visitor in has its own screen (sign-in-recovery.tsx). */
const $signUpItem = computed($signUpRequired, required => (required ? { news: 'v1' } : null))

function SignUpCard({ close, unseen }: TrayCardProps) {
  const copy = useFirstTimeCopy()

  return (
    <TrayCard
      action={{
        label: copy.tray.signUp.action,
        run: () => {
          close()
          startSignUp('tray')
        }
      }}
      description={copy.tray.signUp.description}
      title={copy.tray.signUp.title}
      unseen={unseen}
    />
  )
}

function FirstTimeSplashCard({ close, unseen }: TrayCardProps) {
  const copy = useFirstTimeCopy().tray.meet

  return (
    <TrayCard
      action={{
        label: copy.action,
        run: () => {
          close()
          reopenFirstTimeSplash()
        }
      }}
      description={copy.description}
      title={copy.title}
      unseen={unseen}
    />
  )
}

/** The usage-stats question the sidebar card put away (store/usage-stats-offer.ts): listed until it
 *  is answered or cleared, news once. Review opens the short dialog, which can answer it. */
const $usageStatsItem = computed($usageStatsOfferPlace, place => (place === 'tray' ? { news: 'v1' } : null))

function UsageStatsCard({ close, dismiss, unseen }: TrayCardProps) {
  const copy = useUsageStatsCopy().tray

  return (
    <TrayCard
      action={{
        label: copy.action,
        run: () => {
          close()
          openUsageStatsDetails()
        }
      }}
      clear={{
        label: copy.clear,
        run: () => {
          dismiss()
          clearUsageStatsOffer()
        }
      }}
      description={copy.description}
      title={copy.title}
      unseen={unseen}
    />
  )
}

/** The tray's items, in the order they list. */
export const TRAY_ITEMS: readonly TrayItemDefinition[] = [
  { id: 'sign-up', $state: $signUpItem, Card: SignUpCard },
  { id: 'app-update', $state: $pendingUpdate, Card: AppUpdateCard },
  { id: 'usage-stats', $state: $usageStatsItem, Card: UsageStatsCard },
  {
    id: 'new-version',
    $state: computed($newVersionReady, ready => (ready ? { news: 'ready' } : null)),
    Card: NewVersionCard
  },
  // Once seen, the splash's entry is news once more: the splash says it can be found here.
  {
    id: 'rename-splash',
    $state: computed($renameSplashReopenable, reopenable => (reopenable ? { news: 'v1' } : null)),
    Card: RenameSplashCard
  },
  // The same for the first-time splash, which says it can be found here too.
  {
    id: 'first-time-splash',
    $state: computed($firstTimeSplashReopenable, reopenable => (reopenable ? { news: 'v1' } : null)),
    Card: FirstTimeSplashCard
  }
]

export const notificationTray = createNotificationTray(TRAY_ITEMS)
