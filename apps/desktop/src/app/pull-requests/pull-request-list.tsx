import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import type { HermesGithubPullRequestSummary } from '@/global'

import { relativeTime } from './pull-request-utils'

interface Props {
  items: HermesGithubPullRequestSummary[]
  selected?: string
  onSelect: (item: HermesGithubPullRequestSummary) => void
  onOpen: (url: string) => void
}

export function PullRequestList({ items, selected, onSelect, onOpen }: Props) {
  return (
    <div className="h-full overflow-y-auto" role="listbox">
      {items.map(item => {
        const status = item.isDraft ? 'DRAFT' : item.state

        return (
          <button
            aria-selected={selected === item.id}
            className="group flex w-full min-w-0 items-start gap-3 px-4 py-3 text-left hover:bg-(--chrome-action-hover) aria-selected:bg-(--ui-bg-quaternary)"
            key={item.id}
            onClick={() => onSelect(item)}
            role="option"
          >
            <Codicon className="mt-0.5 shrink-0 text-(--ui-text-tertiary)" name="git-pull-request" />
            <span className="min-w-0 flex-1">
              <span className="flex min-w-0 items-center gap-2 text-xs text-(--ui-text-secondary)">
                <span className="truncate">{item.repository}</span>
                <span className="shrink-0">#{item.number}</span>
                <span className="shrink-0 text-[10px] font-medium">{status}</span>
              </span>
              <span className="mt-1 block truncate text-sm font-medium text-(--ui-text-primary)">{item.title}</span>
              <span className="mt-1 flex min-w-0 items-center gap-2 text-xs text-(--ui-text-tertiary)">
                {item.author && <span className="truncate">@{item.author.login}</span>}
                {item.labels.slice(0, 3).map(label => (
                  <span className="max-w-24 truncate" key={label.name}>
                    {label.name}
                  </span>
                ))}
                {item.labels.length > 3 && <span>+{item.labels.length - 3}</span>}
                <span className="ml-auto shrink-0">
                  <Codicon name="comment" /> {item.commentsCount}
                </span>
                <span className="shrink-0">{relativeTime(item.updatedAt)}</span>
              </span>
            </span>
            <Button
              aria-label={`Open ${item.repository} pull request ${item.number} on GitHub`}
              onClick={event => {
                event.stopPropagation()
                onOpen(item.url)
              }}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="link-external" />
            </Button>
          </button>
        )
      })}
    </div>
  )
}
