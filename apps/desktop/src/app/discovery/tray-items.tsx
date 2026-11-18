import { computed } from 'nanostores'
import type { ComponentType } from 'react'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { $cloudOwnsUpdates, updateTargetFor } from '@/store/cloud-updates'
import { createNotificationTray, type TrayItemSource } from '@/store/notification-tray'
import { $renameSplashReopenable, reopenRenameSplash } from '@/store/rename-splash'
import { $connection } from '@/store/session'
import { $backendUpdateApply, $backendUpdateStatus, $updateApply, $updateStatus, startActiveUpdate } from '@/store/updates'

import { useDiscoveryCopy } from './discovery-copy'

export interface TrayCardProps {
  /** Closes the tray: call it before an action that takes the person somewhere else. */
  close: () => void
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
  unseen: boolean
}

/** The tray's one card shape: a caption, a display title, a sentence, one action. Items compose it
 *  so every entry reads alike; an item that needs a different shape renders its own. */
export function TrayCard({ action, description, eyebrow, title, unseen }: TrayCardViewProps) {
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

/** The tray's items, in the order they list. */
export const TRAY_ITEMS: readonly TrayItemDefinition[] = [
  { id: 'app-update', $state: $pendingUpdate, Card: AppUpdateCard },
  // Once seen, the splash's entry is news once more: the splash says it can be found here.
  {
    id: 'rename-splash',
    $state: computed($renameSplashReopenable, reopenable => (reopenable ? { news: 'v1' } : null)),
    Card: RenameSplashCard
  }
]

export const notificationTray = createNotificationTray(TRAY_ITEMS)
