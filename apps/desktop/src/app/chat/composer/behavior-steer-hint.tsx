import { KbdCombo } from '@/components/ui/kbd'
import { useI18n } from '@/i18n'
import { behaviorKeyAction } from '@/store/composer-behavior'

import type { ComposerBusyAction } from './busy-submit-state'

/** Advertise only a live steer chord; ordinary Send keeps the native busy policy. */
export function BehaviorSteerHint({ canSteer, busyAction }: { canSteer: boolean; busyAction: ComposerBusyAction }) {
  const { t } = useI18n()

  const combo =
    behaviorKeyAction({ key: 'Enter', busy: true, ctrlKey: true, metaKey: true }) === 'steer'
      ? 'mod+enter'
      : busyAction === 'steer'
        ? 'enter'
        : null

  if (!canSteer || !combo) {
    return null
  }

  return (
    <div
      className="flex items-center justify-end gap-1 text-[0.68rem] leading-none text-muted-foreground/70"
      data-slot="composer-steer-hint"
    >
      <KbdCombo combo={combo} size="sm" />
      <span>{t.composer.steer}</span>
    </div>
  )
}
