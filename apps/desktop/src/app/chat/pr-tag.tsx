import { Codicon } from '@/components/ui/codicon'
import { Tip } from '@/components/ui/tooltip'
import type { HermesBranchPullRequest } from '@/global'
import { cn } from '@/lib/utils'
import { pullRequestBucket } from '@/store/pull-requests'

// GitHub's own colour language, mapped onto our tokens: open is the "go" green,
// merged the purple everyone reads as landed, draft and closed muted since
// neither is waiting on you.
const PR_STYLE: Record<string, { className: string; icon: string }> = {
  closed: { className: 'text-(--ui-red)', icon: 'git-pull-request-closed' },
  draft: { className: 'text-(--ui-text-quaternary)', icon: 'git-pull-request-draft' },
  merged: { className: 'text-(--ui-purple)', icon: 'git-merge' },
  open: { className: 'text-(--ui-green)', icon: 'git-pull-request' }
}

export function openPullRequest(pr: HermesBranchPullRequest): void {
  if (pr.url) {
    void window.hermesDesktop?.openExternal?.(pr.url)
  }
}

/** The branch's PR as a row chip: state glyph plus number, tooltipped with the
 *  title, and a link to the PR on click. Identity like {@link ProfileTag} —
 *  never a status dot. */
export function PrTag({ className, pr }: { className?: string; pr: HermesBranchPullRequest }) {
  const style = PR_STYLE[pullRequestBucket(pr)] ?? PR_STYLE.open

  return (
    <Tip label={`#${pr.number} ${pr.title}`}>
      <button
        aria-label={`Open pull request #${pr.number}`}
        // A flex box doesn't pass text-decoration down to its items, so the
        // underline goes on the number itself rather than the chip.
        className={cn(
          'group/pr flex shrink-0 items-center gap-0.5 text-[0.625rem] leading-none tabular-nums',
          style.className,
          className
        )}
        // Marks the chip as a live link for the row's hover rule: while the
        // pointer is on it, the row keeps its metadata and holds the kebab back
        // (see session-row) so the click can actually land.
        data-pr-link
        onClick={event => {
          // The row underneath opens the session on click and pins on
          // shift-click; the chip is its own target and keeps the press.
          event.preventDefault()
          event.stopPropagation()
          openPullRequest(pr)
        }}
        onPointerDown={event => event.stopPropagation()}
        type="button"
      >
        <Codicon name={style.icon} size="0.75rem" />
        <span className="underline-offset-1 group-hover/pr:underline">{pr.number}</span>
      </button>
    </Tip>
  )
}
