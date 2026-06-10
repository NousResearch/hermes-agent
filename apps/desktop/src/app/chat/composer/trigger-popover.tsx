import type { Unstable_TriggerItem } from '@assistant-ui/core'

import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

import {
  COMPLETION_DRAWER_BELOW_CLASS,
  COMPLETION_DRAWER_CLASS,
  COMPLETION_DRAWER_ROW_CLASS,
  CompletionDrawerEmpty
} from './completion-drawer'

const AT_ICON_BY_TYPE: Record<string, string> = {
  diff: 'diff',
  file: 'book',
  folder: 'folder',
  git: 'git-branch',
  image: 'file-media',
  simple: 'symbol-misc',
  staged: 'diff-added',
  tool: 'tools',
  url: 'globe'
}

function completionIcon(kind: '@' | '/', item: Unstable_TriggerItem) {
  if (kind === '/') {
    return 'terminal'
  }

  const meta = item.metadata as { rawText?: string } | undefined
  const raw = meta?.rawText || item.label

  if (raw.startsWith('@diff')) {
    return AT_ICON_BY_TYPE.diff
  }

  if (raw.startsWith('@staged')) {
    return AT_ICON_BY_TYPE.staged
  }

  return AT_ICON_BY_TYPE[item.type] || AT_ICON_BY_TYPE.simple
}

interface ComposerTriggerPopoverProps {
  activeIndex: number
  items: readonly Unstable_TriggerItem[]
  kind: '@' | '/'
  loading: boolean
  onHover: (index: number) => void
  onPick: (item: Unstable_TriggerItem) => void
  placement?: 'bottom' | 'top'
}

export function ComposerTriggerPopover({
  activeIndex,
  items,
  kind,
  loading,
  onHover,
  onPick,
  placement = 'top'
}: ComposerTriggerPopoverProps) {
  const { t } = useI18n()
  const copy = t.composer

  return (
    <div
      className={placement === 'bottom' ? COMPLETION_DRAWER_BELOW_CLASS : COMPLETION_DRAWER_CLASS}
      data-slot="composer-completion-drawer"
      data-state="open"
      onMouseDown={event => event.preventDefault()}
      role="listbox"
    >
      {items.length === 0 ? (
        <CompletionDrawerEmpty title={loading ? copy.lookupLoading : copy.lookupNoMatches}>
          {kind === '@' ? (
            <>
              {copy.lookupTry} <span className="font-mono text-foreground/80">@file:</span> {copy.lookupOr}{' '}
              <span className="font-mono text-foreground/80">@folder:</span>.
            </>
          ) : (
            <>
              {copy.lookupTry} <span className="font-mono text-foreground/80">/help</span>.
            </>
          )}
        </CompletionDrawerEmpty>
      ) : (
        items.map((item, index) => {
          const meta = item.metadata as { display?: string; meta?: string } | undefined
          const display = meta?.display ?? (kind === '/' ? `/${item.label}` : item.label)
          const description = meta?.meta || item.description

          return (
            <button
              className={cn(COMPLETION_DRAWER_ROW_CLASS, index === activeIndex && 'bg-(--ui-bg-tertiary)')}
              data-highlighted={index === activeIndex ? '' : undefined}
              key={item.id}
              onClick={() => onPick(item)}
              onMouseEnter={() => onHover(index)}
              type="button"
            >
              <span className="grid size-3.5 shrink-0 place-items-center text-(--ui-text-tertiary)">
                <Codicon name={completionIcon(kind, item)} size="0.875rem" />
              </span>
              <span className="min-w-0 shrink truncate font-mono font-medium leading-5 text-foreground">{display}</span>
              {description && (
                <span className="min-w-0 flex-1 truncate leading-5 text-(--ui-text-tertiary)">{description}</span>
              )}
            </button>
          )
        })
      )}
    </div>
  )
}
