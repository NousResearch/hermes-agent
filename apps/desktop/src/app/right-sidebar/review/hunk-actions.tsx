import { useStore } from '@nanostores/react'

import type { DiffHunkRef } from '@/components/chat/diff-lines'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { Tip } from '@/components/ui/tooltip'
import type { HermesReviewFile } from '@/global'
import { useI18n } from '@/i18n'
import { displayPath } from '@/lib/display-path'
import { notifyError } from '@/store/notifications'
import { $reviewRevertTarget, applyReviewHunk, cancelRevert, confirmRevert, requestRevert } from '@/store/review'

const ACTION_BTN = 'size-5'

/** The per-change row renderer for `file`'s diff, or undefined when its diff has
 *  no hunk git can apply (read-only scopes, an untracked file's all-add diff). */
export function hunkActionsFor(
  file: HermesReviewFile | undefined,
  uncommitted: boolean
): ((hunk: DiffHunkRef) => React.ReactNode) | undefined {
  if (!uncommitted || !file || file.status === '?') {
    return undefined
  }

  return hunk => <ReviewHunkActions hunk={hunk.text} path={file.path} staged={file.staged} />
}

// The small row above each change in the selected file's diff: stage or revert
// that one change (unstage it, in a staged diff). Revert confirms first.
function ReviewHunkActions({ hunk, path, staged }: { hunk: string; path: string; staged: boolean }) {
  const c = useI18n().t.statusStack.coding
  const stageLabel = staged ? c.unstageHunk : c.stageHunk

  return (
    <>
      <Tip label={stageLabel}>
        <Button
          aria-label={stageLabel}
          className={ACTION_BTN}
          onClick={() =>
            void applyReviewHunk(path, hunk, staged ? 'unstage' : 'stage').catch(err => notifyError(err, stageLabel))
          }
          size="icon-xs"
          variant="ghost"
        >
          <Codicon name={staged ? 'remove' : 'add'} size="0.75rem" />
        </Button>
      </Tip>
      {!staged && (
        <Tip label={c.revertHunk}>
          <Button
            aria-label={c.revertHunk}
            className={ACTION_BTN}
            onClick={() => requestRevert(path, hunk)}
            size="icon-xs"
            variant="ghost"
          >
            <Codicon name="discard" size="0.75rem" />
          </Button>
        </Tip>
      )}
    </>
  )
}

/** The destructive confirm for revert all / one file / one change of a file. */
export function ReviewRevertDialog() {
  const c = useI18n().t.statusStack.coding
  const target = useStore($reviewRevertTarget)
  const all = target?.path == null
  const label = all ? c.revertAll : target?.hunk ? c.revertHunk : c.revert
  const prompt = all ? c.revertAllConfirm : target?.hunk ? c.revertHunkConfirm : c.revertConfirm

  return (
    <ConfirmDialog
      confirmLabel={label}
      description={
        <>
          {prompt}
          {!all && target?.path && (
            <span
              className="mt-2 block truncate font-mono text-[0.7rem] text-(--ui-text-secondary)"
              title={displayPath(target.path)}
            >
              {displayPath(target.path)}
            </span>
          )}
        </>
      }
      destructive
      // confirmRevert closes the dialog itself, then reverts in the
      // background — so the failure lands in a toast, not inline.
      dismissOnConfirm
      onClose={cancelRevert}
      onConfirm={() => confirmRevert().catch(err => void notifyError(err, c.revert))}
      open={target !== undefined}
      title={label}
    />
  )
}
