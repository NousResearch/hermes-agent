import { useStore } from '@nanostores/react'
import { useMemo } from 'react'

import { contentFingerprint } from '@/app/review/annotations/anchors'
import { ReviewAnnotationsList } from '@/app/review/annotations/list'
import { PlanReviews } from '@/app/review/annotations/plan-reviews'
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
import { SegmentedControl } from '@/components/ui/segmented-control'
import { Tip } from '@/components/ui/tooltip'
import { useDelayedTrue } from '@/hooks/use-delayed-true'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $annotationContext,
  $annotationDraft,
  $annotationEditorCollapsed,
  $annotations,
  activateAnnotationContext,
  beginAnnotation,
  type DiffAnnotationAnchor,
  reopenAnnotationEditor
} from '@/store/annotations'
import { $panesFlipped } from '@/store/layout'
import { notifyError } from '@/store/notifications'
import {
  $reviewAnnotationContext,
  $reviewDiff,
  $reviewDiffLoading,
  $reviewFiles,
  $reviewIsRepo,
  $reviewLastTurnBaseRef,
  $reviewLoading,
  $reviewRevertTarget,
  $reviewScope,
  $reviewSelectedPath,
  $reviewTreeMode,
  cancelRevert,
  clearReviewSelection,
  closeReview,
  confirmRevert,
  refreshReview,
  requestRevert,
  setReviewScope,
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
  const scope = useStore($reviewScope)
  const lastTurnBaseRef = useStore($reviewLastTurnBaseRef)
  const reviewAnnotationContext = useStore($reviewAnnotationContext)
  const annotations = useStore($annotations)
  const annotationContext = useStore($annotationContext)
  const annotationDraft = useStore($annotationDraft)
  const annotationCollapsed = useStore($annotationEditorCollapsed)

  const selectedFile = files.find(file => file.path === selectedPath)
  const activeReviewContext = reviewAnnotationContext ?? annotationContext

  const selectedFileHasDraft = Boolean(
    selectedFile &&
    annotationCollapsed &&
    annotationDraft?.contextId === activeReviewContext.id &&
    annotationDraft.anchor.path === selectedFile.path
  )

  const scopeOptions = useMemo(
    () => [
      { id: 'uncommitted' as const, label: t.desktop.annotations.scopeUncommitted },
      { id: 'lastTurn' as const, label: t.desktop.annotations.scopeLastTurn },
      { id: 'branch' as const, label: t.desktop.annotations.scopeBranch }
    ],
    [t.desktop.annotations.scopeBranch, t.desktop.annotations.scopeLastTurn, t.desktop.annotations.scopeUncommitted]
  )

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

      <PlanReviews />

      {isRepo && (
        <div className="px-2.5 py-1.5">
          <SegmentedControl onChange={setReviewScope} options={scopeOptions} value={scope} />
        </div>
      )}

      {loading || isRepo ? (
        hasFiles ? (
          <div
            className="contents"
            onPointerDownCapture={() => {
              if (reviewAnnotationContext) {
                activateAnnotationContext(reviewAnnotationContext, { carryStale: true })
              }
            }}
          >
            <ReviewFileTree />
          </div>
        ) : showTreeSkeleton ? (
          <TreeSkeleton />
        ) : loading ? (
          <div className="min-h-0 flex-1" />
        ) : (
          <PaneEmptyState
            label={
              scope === 'lastTurn' && !lastTurnBaseRef ? t.desktop.annotations.lastTurnTracking : t.rightSidebar.noDiffs
            }
          />
        )
      ) : (
        // No repo at all → same terse empty state, just without the chrome.
        <PaneEmptyState label={t.rightSidebar.noDiffs} />
      )}

      {/* Selected file's diff — reuses the shiki-highlighted FileDiffPanel. */}
      {selectedFile && (
        <div
          className="flex max-h-[55%] shrink-0 flex-col border-t border-(--ui-stroke-secondary)"
          onPointerDownCapture={() => {
            if (reviewAnnotationContext) {
              activateAnnotationContext(reviewAnnotationContext, { carryStale: true })
            }
          }}
        >
          <div className="flex items-center gap-1 px-2.5 py-1.5" data-suppress-pane-reveal-side="">
            <span
              className="min-w-0 flex-1 truncate font-mono text-[0.66rem] text-(--ui-text-secondary)"
              title={selectedFile.path}
            >
              {selectedFile.path}
            </span>
            <DiffCount added={selectedFile.added} className="text-[0.64rem] leading-4" removed={selectedFile.removed} />
            <Tip label={selectedFileHasDraft ? t.desktop.annotations.reopen : t.desktop.annotations.add}>
              <Button
                aria-label={selectedFileHasDraft ? t.desktop.annotations.reopen : t.desktop.annotations.add}
                onClick={() =>
                  selectedFileHasDraft
                    ? reopenAnnotationEditor()
                    : beginAnnotation({ kind: 'file', path: selectedFile.path }, null, activeReviewContext)
                }
                size="icon-xs"
                variant="ghost"
              >
                <Codicon name={selectedFileHasDraft ? 'edit' : 'comment'} />
              </Button>
            </Tip>
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
              <FileDiffPanel
                annotateLineLabel={line => `${t.desktop.annotations.add}: ${line}`}
                annotations={annotations
                  .filter(
                    item =>
                      item.status !== 'stale' &&
                      item.status !== 'orphaned' &&
                      item.anchor.kind === 'diff' &&
                      item.anchor.path === selectedFile.path
                  )
                  .map(item => item.anchor as DiffAnnotationAnchor)}
                diff={diff}
                onAnnotateLine={({ editorAnchor, excerpt, line, side }) =>
                  beginAnnotation(
                    {
                      baseRef: activeReviewContext.baseRef,
                      contentHash: contentFingerprint(diff),
                      excerpt,
                      headSha: activeReviewContext.headSha,
                      kind: 'diff',
                      lineEnd: line,
                      lineStart: line,
                      path: selectedFile.path,
                      side
                    },
                    editorAnchor,
                    activeReviewContext
                  )
                }
                path={selectedFile.path}
                showLineNumbers
              />
            ) : (
              <div className="py-6 text-center text-[0.66rem] text-muted-foreground/60">{c.noDiff}</div>
            )}
          </div>
        </div>
      )}

      <ReviewAnnotationsList />

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
