import { useStore } from '@nanostores/react'

import { composerFloatingPill } from '@/components/chat/composer-dock'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $preloadingModel } from '@/store/local-model-preload'

/** While the backend loads the local model for the chat being typed, say so above the composer. */
export function ModelPreloadHint() {
  const { t } = useI18n()
  const model = useStore($preloadingModel)

  if (!model) {
    return null
  }

  return (
    // Informational, not a control — but it keeps pointer events: the pop-out drag region sits
    // behind the strip (see micro-actions.tsx).
    <span
      className={cn(composerFloatingPill, 'cursor-default hover:bg-(--composer-fill) hover:text-(--ui-text-secondary)')}
      data-slot="composer-model-preload"
      role="status"
    >
      <span className="truncate">{t.composer.loadingLocalModel(model)}</span>
    </span>
  )
}
