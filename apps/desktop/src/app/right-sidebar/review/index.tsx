import { useStore } from '@nanostores/react'

import { FileDiffPanel } from '@/components/chat/diff-lines'
import { DiffSkeleton, TreeSkeleton } from '@/components/chat/skeletons'
import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { DiffCount } from '@/components/ui/diff-count'
import { Tip } from '@/components/ui/tooltip'
import { useDelayedTrue } from '@/hooks/use-delayed-true'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { $panesFlipped } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import {
  $reviewDiff,
  $reviewDiffLoading,
  $reviewFiles,
  $reviewIsRepo,
  $reviewLoading,
  $reviewRevertTarget,
  $reviewSelectedPath,
  $reviewTreeMode,
  cancelRevert,
  clearReviewSelection,
  closeReview,
  confirmRevert,
  refreshReview,
  requestRevert,
  stageReviewFile,
  toggleReviewTreeMode,
  unstageReviewFile
} from '@/store/review'

import { SidebarPanelLabel } from '../../shell/sidebar-label'
import { PaneEmptyState, RightSidebarSectionHeader } from '../index'

import { ReviewFileTree } from './file-tree'
import { ReviewShipBar } from './ship-bar'

// Compact header/diff action buttons — micro hit targets packed tight, matching
// the rest of the app's icon-action rows.
const ACTION_BTN = 'size-5'

export function ReviewPane() {
  const { t } = useI18n()
  const c = t.statusStack.coding
  const panesFlipped = useStore($panesFlipped)
  const files = useStore($reviewFiles)
  const loading = useStore($reviewLoading)
  const isRepo = useStore($reviewIsRepo)
  const selectedPath = useStore($reviewSelectedPath)
  const diff = useStore($reviewDiff)
  const diffLoading = useStore($reviewDiffLoading)
  const revertTarget = useStore($reviewRevertTarget)
  const treeMode = useStore($reviewTreeMode)

  const selectedFile = files.find(file => file.path === selectedPath)
  const hasFiles = files.length > 0
  // `{ path: null }` → revert all; `{ path: '…' }` → revert one file.
  const revertingAll = revertTarget?.path == null
  // Delay the skeletons so fast loads (most project switches) just blank → content
  // instead of flashing a jarring loading state.
  const showTreeSkeleton = useDelayedTrue(loading && !hasFiles)
  const showDiffSkeleton = useDelayedTrue(diffLoading)

  return (
    <aside
      aria-label={c.review}
      className={cn(
        'before:pointer-events-none relative flex h-full w-full min-w-0 flex-col overflow-hidden border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) pt-(--titlebar-height) text-(--ui-text-tertiary)',
        panesFlipped
          ? 'border-r shadow-[inset_-0.0625rem_0_0_color-mix(in_srgb,white_18%,transparent)]'
          : 'border-l shadow-[inset_0.0625rem_0_0_color-mix(in_srgb,white_18%,transparent)]'
      )}
    >
      {(loading || isRepo) && (
        <RightSidebarSectionHeader data-suppress-pane-reveal-side="">
          <div className="flex min-w-0 flex-1">
            {/* Pure self-naming label — redundant under a zone tab that already
                says "review", so the zone header hides it (styles.css). */}
            <SidebarPanelLabel data-pane-self-label="">{c.review}</SidebarPanelLabel>
          </div>
          <Tip label={treeMode === 'tree' ? c.viewAsList : c.viewAsTree}>
            <Button
              aria-label={treeMode === 'tree' ? c.viewAsList : c.viewAsTree}
              className={ACTION_BTN}
              disabled={!hasFiles}
              onClick={toggleReviewTreeMode}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name={treeMode === 'tree' ? 'list-flat' : 'list-tree'} size="0.8125rem" />
            </Button>
          </Tip>
          <Tip label={c.stageAll}>
            <Button
              aria-label={c.stageAll}
              className={ACTION_BTN}
              disabled={!hasFiles}
              onClick={() => void stageReviewFile(null).catch(err => notifyError(err, c.stageAll))}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="add" size="0.8125rem" />
            </Button>
          </Tip>
          <Tip label={c.revertAll}>
            <Button
              aria-label={c.revertAll}
              className={ACTION_BTN}
              disabled={!hasFiles}
              onClick={() => requestRevert(null)}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="discard" size="0.8125rem" />
            </Button>
          </Tip>
          <Tip label={t.rightSidebar.refreshTree}>
            <Button
              aria-label={t.rightSidebar.refreshTree}
              className={ACTION_BTN}
              onClick={() => void refreshReview()}
              size="icon-xs"
              variant="ghost"
            >
              <Codicon name="refresh" size="0.8125rem" spinning={loading} />
            </Button>
          </Tip>
          <Tip label={c.close}>
            <Button aria-label={c.close} className={ACTION_BTN} onClick={closeReview} size="icon-xs" variant="ghost">
              <Codicon name="close" size="0.8125rem" />
            </Button>
          </Tip>
        </RightSidebarSectionHeader>
      )}

      {loading || isRepo ? (
        hasFiles ? (
          <ReviewFileTree />
        ) : showTreeSkeleton ? (
          <TreeSkeleton />
        ) : loading ? (
          <div className="min-h-0 flex-1" />
        ) : (
          <PaneEmptyState label={t.rightSidebar.noDiffs} />
        )
      ) : (
        // No repo at all → same terse empty state, just without the chrome.
        <PaneEmptyState label={t.rightSidebar.noDiffs} />
      )}

      {/* Selected file's diff — reuses the shiki-highlighted FileDiffPanel. */}
      {selectedFile && (
        <div className="flex max-h-[55%] shrink-0 flex-col border-t border-(--ui-stroke-secondary)">
          <div className="flex items-center gap-1 px-2.5 py-1.5" data-suppress-pane-reveal-side="">
            <span
              className="min-w-0 flex-1 truncate font-mono text-[0.66rem] text-(--ui-text-secondary)"
              title={selectedFile.path}
            >
              {selectedFile.path}
            </span>
            <DiffCount added={selectedFile.added} className="text-[0.64rem] leading-4" removed={selectedFile.removed} />
            <Tip label={selectedFile.staged ? c.unstage : c.stage}>
              <Button
                aria-label={selectedFile.staged ? c.unstage : c.stage}
                className={ACTION_BTN}
                onClick={() =>
                  void (
                    selectedFile.staged ? unstageReviewFile(selectedFile.path) : stageReviewFile(selectedFile.path)
                  ).catch(err => notifyError(err, c.stage))
                }
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name={selectedFile.staged ? 'remove' : 'add'} size="0.8rem" />
              </Button>
            </Tip>
            <Tip label={c.close}>
              <Button
                aria-label={c.close}
                className={ACTION_BTN}
                onClick={clearReviewSelection}
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name="close" size="0.8rem" />
              </Button>
            </Tip>
          </div>
          <div className="min-h-0 flex-1 overflow-auto px-1 pb-1">
            {diffLoading ? (
              showDiffSkeleton ? (
                <DiffSkeleton />
              ) : null
            ) : diff ? (
              <FileDiffPanel diff={diff} path={selectedFile.path} />
            ) : (
              <div className="py-6 text-center text-[0.66rem] text-muted-foreground/60">{c.noDiff}</div>
            )}
          </div>
        </div>
      )}

      <ReviewShipBar />

      <Dialog onOpenChange={open => !open && cancelRevert()} open={revertTarget !== undefined}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{revertingAll ? c.revertAll : c.revert}</DialogTitle>
            <DialogDescription>
              {revertingAll ? c.revertAllConfirm : c.revertConfirm}
              {!revertingAll && revertTarget?.path && (
                <span
                  className="mt-2 block truncate font-mono text-[0.7rem] text-(--ui-text-secondary)"
                  title={revertTarget.path}
                >
                  {revertTarget.path}
                </span>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button onClick={cancelRevert} variant="ghost">
              {t.common.cancel}
            </Button>
            <Button onClick={() => void confirmRevert().catch(err => notifyError(err, c.revert))} variant="destructive">
              {revertingAll ? c.revertAll : c.revert}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  )
}
