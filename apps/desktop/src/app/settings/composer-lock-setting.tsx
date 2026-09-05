import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { $composerPopoutGesturesEnabled, setComposerPopoutGesturesEnabled } from '@/store/composer-popout'

import { ToggleRow } from './primitives'

/** Device-local lock: docked composer cannot peel out. Shared by Chat + Appearance. */
export function ComposerLockSetting({ id }: { id?: string }) {
  const { t } = useI18n()
  const a = t.settings.appearance
  const gesturesEnabled = useStore($composerPopoutGesturesEnabled)
  const row = (
    <ToggleRow
      checked={!gesturesEnabled}
      description={a.composerPopoutDesc}
      label={a.composerPopoutTitle}
      onChange={locked => setComposerPopoutGesturesEnabled(!locked)}
    />
  )

  if (!id) {
    return row
  }

  return (
    <div className="scroll-mt-6" id={id}>
      {row}
    </div>
  )
}
