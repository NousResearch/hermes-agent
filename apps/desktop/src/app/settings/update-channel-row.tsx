import { useStore } from '@nanostores/react'
import type { ReactElement } from 'react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { confirm } from '@/store/confirm'
import { notifyError } from '@/store/notifications'
import { $updateChecking, $updateStatus, setUpdateChannel, type SourceUpdateChannel, sourceUpdateChannel } from '@/store/updates'

import { ListRow } from './primitives'

/**
 * Stable releases vs every commit on main, for a source checkout. Packaged
 * installs (Store, MSIX, bundled) bake their channel and never render this.
 */
export function UpdateChannelRow(): ReactElement | null {
  const { t } = useI18n()
  const a = t.settings.about.channel
  const status = useStore($updateStatus)
  const checking = useStore($updateChecking)
  const channel = sourceUpdateChannel(status)

  if (channel === null) {
    return null
  }

  const change = async (next: SourceUpdateChannel): Promise<void> => {
    if (next === channel) {
      return
    }

    // Main → stable can land on an older commit than the one running now.
    if (
      next === 'stable' &&
      !(await confirm({
        title: a.stableConfirmTitle,
        description: a.stableConfirmBody,
        confirmLabel: a.stableConfirm
      }))
    ) {
      return
    }

    triggerHaptic('crisp')

    try {
      await setUpdateChannel(next)
    } catch (error) {
      notifyError(error, a.failed)
    }
  }

  return (
    <ListRow
      action={
        <SegmentedControl
          disabled={checking}
          onChange={id => void change(id)}
          options={[
            { id: 'stable', label: a.stable },
            { id: 'main', label: a.main }
          ]}
          value={channel}
        />
      }
      description={a.description}
      title={a.title}
    />
  )
}
