import { useStore } from '@nanostores/react'

import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $hudMode } from '@/store/hud'
import { $showsAdvancedChrome } from '@/store/interface-mode'

interface ComposerLayoutInput {
  compactPill: boolean
  guidedChat: boolean
  stacked: boolean
}

/**
 * The composer's row layout. Simple mode always uses two rows — text above,
 * context and controls below — with Send in place and the full model control,
 * outside the HUD and the setup guide. Advanced keeps the width-driven inline
 * and stacked layouts.
 */
export function useComposerLayout({ compactPill, guidedChat, stacked }: ComposerLayoutInput) {
  const { t } = useI18n()
  const showsAdvancedChrome = useStore($showsAdvancedChrome)
  const hudMode = useStore($hudMode)
  const simple = !showsAdvancedChrome && !hudMode && !guidedChat

  return {
    compactPill: compactPill && !simple,
    gridClass: cn(
      'grid w-full',
      simple
        ? 'grid-cols-[minmax(0,1fr)_auto] items-center gap-(--composer-row-gap) [grid-template-areas:"input_input"_"menu_controls"]'
        : stacked
          ? 'grid-cols-[auto_1fr] gap-(--composer-row-gap) [grid-template-areas:"input_input"_"menu_controls"]'
          : 'grid-cols-[auto_1fr_auto] items-center gap-(--composer-control-gap) [grid-template-areas:"menu_input_controls"]'
    ),
    menuClass: cn(
      'flex min-w-0 gap-(--composer-control-gap) [grid-area:menu]',
      simple ? 'items-center' : 'translate-y-[3px] items-start self-start'
    ),
    placeholder: (disabled: boolean, placeholder: string) => (simple && !disabled ? t.composer.message : placeholder),
    simple,
    stackedInput: stacked || simple
  }
}
