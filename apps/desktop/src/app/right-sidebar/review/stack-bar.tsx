import { useStore } from '@nanostores/react'

import { FileDiffPanel } from '@/components/chat/diff-lines'
import { DiffSkeleton } from '@/components/chat/skeletons'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { DiffCount } from '@/components/ui/diff-count'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { useDelayedTrue } from '@/hooks/use-delayed-true'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import { $reviewFiles, $reviewShipBusy } from '@/store/review'
import {
  $reviewRestackPending,
  $reviewStackCommits,
  $reviewStackDiff,
  $reviewStackDiffLoading,
  $reviewStackLoading,
  $reviewStackSelected,
  $reviewStackSelectedCommit,
  clearStackSelection,
  requestRestack,
  selectStackCommit
} from '@/store/review-stack'

const ICON = '0.8rem'
const WORKING_TREE = '__working_tree__'

// The commit-stack row of the review pane: "Restack" hands the agent the job of
// rewriting this branch's commits into reviewable pieces (same tree, cleaner
// history); the picker beside it walks the resulting commits one patch at a
// time. Hidden when the branch has no commits of its own and a clean tree —
// there is nothing to restack or to walk.
export function ReviewStackBar() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const commits = useStore($reviewStackCommits)
  const loading = useStore($reviewStackLoading)
  const selected = useStore($reviewStackSelected)
  const selectedCommit = useStore($reviewStackSelectedCommit)
  const diff = useStore($reviewStackDiff)
  const diffLoading = useStore($reviewStackDiffLoading)
  const pending = useStore($reviewRestackPending)
  const shipBusy = useStore($reviewShipBusy)
  const files = useStore($reviewFiles)
  const showDiffSkeleton = useDelayedTrue(diffLoading)

  const dirty = files.length > 0
  const hasCommits = commits.length > 0

  if (!hasCommits && !dirty) {
    return null
  }

  const runRestack = () => {
    const outcome = requestRestack(
      {
        intro: c.restackPromptIntro,
        rules: c.restackPromptRules,
        commitLine: (short, subject) => `- ${short} ${subject}`,
        dirtyNote: c.restackPromptDirty
      },
      dirty
    )

    if (outcome === 'unavailable') {
      notifyError(new Error(c.agentShipUnavailable), c.restack)
    } else if (outcome === 'nothing') {
      notify({ kind: 'info', message: c.restackNothing })
    }
  }

  return (
    <div className="flex shrink-0 flex-col border-t border-(--ui-stroke-secondary)" data-suppress-pane-reveal-side="">
      <div className="flex items-center gap-1 px-2 py-1.5">
        <Tip label={pending ? c.restackPending : c.restackTip}>
          <Button
            aria-label={c.restack}
            className="h-6 gap-1 px-2 text-[0.7rem]"
            disabled={pending || shipBusy || loading}
            onClick={runRestack}
            size="sm"
            variant="outline"
          >
            <Codicon name="git-commit" size={ICON} spinning={pending} />
            <span>{c.restack}</span>
          </Button>
        </Tip>

        {hasCommits && (
          <Select
            onValueChange={value => void selectStackCommit(value === WORKING_TREE ? null : value)}
            value={selected ?? WORKING_TREE}
          >
            <SelectTrigger aria-label={c.commitPicker} className="h-6 min-w-0 flex-1 px-2 text-[0.7rem]" size="sm">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={WORKING_TREE}>{c.commitPickerWorkingTree}</SelectItem>
              {commits.map((commit, index) => (
                <SelectItem key={commit.sha} value={commit.sha}>
                  <span className="flex min-w-0 items-baseline gap-1.5">
                    <span className="shrink-0 font-mono text-[0.66rem] text-(--ui-text-tertiary)">
                      {index + 1}/{commits.length}
                    </span>
                    <span className="min-w-0 truncate">{commit.subject}</span>
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      {/* The selected commit's patch: header (sha, files, churn) + diff. */}
      {selectedCommit && (
        <div className="flex max-h-[55%] min-h-0 flex-col">
          <div className="flex items-center gap-1.5 px-2.5 py-1">
            <span className="shrink-0 font-mono text-[0.66rem] text-(--ui-text-tertiary)">{selectedCommit.short}</span>
            <span
              className="min-w-0 flex-1 truncate text-[0.66rem] text-(--ui-text-secondary)"
              title={selectedCommit.subject}
            >
              {selectedCommit.subject}
            </span>
            <span className="shrink-0 text-[0.64rem] text-(--ui-text-tertiary)">
              {c.commitFiles(selectedCommit.files.length)}
            </span>
            <DiffCount
              added={selectedCommit.added}
              className="text-[0.64rem] leading-4"
              removed={selectedCommit.removed}
            />
            <Button aria-label={c.close} className="size-5" onClick={clearStackSelection} size="icon-xs" variant="ghost">
              <Codicon name="close" size={ICON} />
            </Button>
          </div>
          <div className="min-h-0 flex-1 overflow-auto px-1 pb-1">
            {diffLoading ? (
              showDiffSkeleton ? (
                <DiffSkeleton />
              ) : null
            ) : diff ? (
              <FileDiffPanel className="mx-0 mb-0 h-full max-h-none" diff={diff} path={selectedCommit.short} virtualized />
            ) : (
              <div className="py-6 text-center text-[0.66rem] text-muted-foreground/60">{c.noDiff}</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
