/**
 * Skill strip view — renders the transient suggestion hint above the
 * composer. Purely presentational: takes `items` from `useSkillStrip` and
 * renders one pill per skill (command + description) plus a dismiss control.
 * Auto-hides by itself via the hook's timer, so the view is a simple map.
 */
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import type { SkillStripItem } from './use-skill-strip'

export interface SkillStripViewProps {
  /** Skill pills to render. Empty hides the strip entirely. */
  items: SkillStripItem[]
  /** Called when the user clicks 不再显示 — disables the strip forever. */
  onDismissForever: () => void
  className?: string
}

export function SkillStripView({ items, onDismissForever, className }: SkillStripViewProps) {
  const { t } = useI18n()

  if (items.length === 0) {
    return null
  }

  return (
    <div
      className={cn(
        'pointer-events-auto mb-1 flex flex-wrap items-center gap-1.5',
        'animate-[fade-in_0.15s_ease-out] text-[0.72rem] text-(--ui-text-tertiary)',
        className
      )}
      data-slot="composer-skill-strip"
    >
      <span className="opacity-80">{t.composer.skillStripPrefix}</span>
      {items.map(item => (
        <span
          className="inline-flex items-center gap-1 rounded-full border border-(--ui-border-subtle) bg-(--ui-surface-secondary)/60 px-2 py-0.5"
          key={item.command}
          title={item.description}
        >
          <span className="font-mono italic opacity-90">{item.command}</span>
          {item.description ? (
            <span className="whitespace-normal opacity-70">— {item.description}</span>
          ) : null}
        </span>
      ))}
      <button
        aria-label={t.composer.skillStripDismiss}
        className="ml-auto rounded px-1.5 py-0.5 opacity-50 transition-opacity hover:opacity-100"
        onClick={onDismissForever}
        type="button"
      >
        {t.composer.skillStripDismiss}
      </button>
    </div>
  )
}
