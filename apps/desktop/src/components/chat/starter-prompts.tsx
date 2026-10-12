import { requestComposerSetDraft } from '@/app/chat/composer/focus'
import { useI18n } from '@/i18n'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { freshDraftScope } from '@/store/composer'

const STARTERS = ['email', 'dinners', 'summarize', 'lookup'] as const

/**
 * Simple mode's empty-chat suggestions. A card OFFERS a prompt: it fills the
 * new-chat composer (addressed by the fresh-draft key, so a hidden tile never
 * receives it) and the composer takes focus — the user still decides to send.
 * The intro around it is `pointer-events-none`; only the cards opt back in.
 */
export function StarterPrompts() {
  const { t } = useI18n()
  const copy = t.chatStarters

  return (
    <div
      aria-label={copy.label}
      className="pointer-events-auto mx-auto mt-6 grid w-full max-w-xl grid-cols-1 gap-2 sm:grid-cols-2"
      data-slot="aui_starter-prompts"
      role="group"
    >
      {STARTERS.map(id => (
        <button
          className={cn(selectableCardClass({ prominent: true }), 'flex min-w-0 flex-col gap-0.5 px-3 py-2 text-left')}
          key={id}
          onClick={() => void requestComposerSetDraft([freshDraftScope()], copy[id].prompt)}
          type="button"
        >
          <span className="truncate text-sm text-(--ui-text-primary)">{copy[id].title}</span>
          <span className="line-clamp-2 text-xs text-(--ui-text-tertiary)">{copy[id].prompt}</span>
        </button>
      ))}
    </div>
  )
}
