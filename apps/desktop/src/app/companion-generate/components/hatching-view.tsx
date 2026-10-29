import { CompanionEggHatch } from '@/components/companion/companion-egg-hatch'
import { useI18n } from '@/i18n'
import { cancelHatch, type CompanionHatchStage } from '@/store/companion-generate'

interface HatchingViewProps {
  stage: CompanionHatchStage | null
}

// The hatch progress screen — a beating egg with a phase-tracking subtitle
// (per-row → composing → saving).
export function HatchingView({ stage }: HatchingViewProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generateCompanion

  const subtitle = stage
    ? stage.phase === 'row'
      ? copy.hatchRow(stage.state ?? '', stage.done ?? 0, stage.total ?? 0)
      : stage.phase === 'compose'
        ? copy.hatchComposing
        : copy.hatchSaving
    : copy.hatchingSub

  return <CompanionEggHatch cancelLabel={t.common.cancel} onCancel={cancelHatch} subtitle={subtitle} />
}
