import { useStore } from '@nanostores/react'

import { Codicon } from '@/components/ui/codicon'
import { EmptyState } from '@/components/ui/empty-state'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import { confirm } from '@/store/confirm'

import { workspaceBasename } from '../../state'

import { IdeFileEditor } from './file-editor'
import { $ideDirtyPaths, $ideEditor, activateIdeFile, closeIdeFile } from './tabs'

/**
 * Editor column: the open-file tab strip plus the active file's CodeMirror
 * surface. Closing a dirty tab asks first — this is the only place unsaved
 * edits can be discarded by a single click.
 */
export function EditorRegion() {
  const { t } = useI18n()
  const editor = useStore($ideEditor)
  const dirtyPaths = useStore($ideDirtyPaths)

  const requestClose = async (path: string) => {
    if (dirtyPaths.includes(path)) {
      const ok = await confirm({
        confirmLabel: t.ide.closeDirtyConfirm,
        description: t.ide.closeDirtyBody,
        destructive: true,
        title: t.ide.closeDirtyTitle
      })

      if (!ok) {
        return
      }
    }

    closeIdeFile(path)
  }

  return (
    <section
      aria-label={t.ide.editorTitle}
      className="flex h-full min-h-0 w-full min-w-0 flex-col bg-(--ui-chat-surface-background)"
    >
      {editor.openPaths.length > 0 && (
        <div
          aria-label={t.ide.editorTabsLabel}
          className="flex h-9 shrink-0 items-stretch gap-0.5 overflow-x-auto border-b border-(--ui-stroke-tertiary) px-1 pt-1"
          role="tablist"
        >
          {editor.openPaths.map(path => {
            const active = path === editor.activePath
            const dirty = dirtyPaths.includes(path)

            return (
              <div
                aria-selected={active}
                className={cn(
                  'group flex h-8 max-w-56 min-w-0 shrink-0 items-center gap-1.5 rounded-t-sm border border-b-0 border-transparent px-2 text-xs',
                  active
                    ? 'border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) text-foreground'
                    : 'text-(--ui-text-secondary) hover:bg-(--ui-bg-quaternary)'
                )}
                key={path}
                role="tab"
              >
                <button
                  className="flex min-w-0 items-center gap-1.5"
                  onClick={() => activateIdeFile(path)}
                  type="button"
                >
                  <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="file" size={13} />
                  <span className="truncate">{workspaceBasename(path)}</span>
                  {dirty && <span aria-hidden className="size-1.5 shrink-0 rounded-full bg-(--ui-accent)" />}
                </button>
                <Tip label={t.ide.closeTab}>
                  <button
                    aria-label={t.ide.closeTabLabel(workspaceBasename(path) ?? path)}
                    className={cn(
                      'grid size-4 shrink-0 place-items-center rounded-sm text-(--ui-text-tertiary) hover:bg-(--ui-bg-quaternary) hover:text-foreground',
                      !dirty && !active && 'opacity-0 group-hover:opacity-100'
                    )}
                    onClick={() => void requestClose(path)}
                    type="button"
                  >
                    <Codicon name="close" size={12} />
                  </button>
                </Tip>
              </div>
            )
          })}
        </div>
      )}
      <div className="min-h-0 flex-1 overflow-hidden">
        {editor.activePath ? (
          <IdeFileEditor key={editor.activePath} path={editor.activePath} />
        ) : (
          <EmptyState description={t.ide.editorEmptyBody} title={t.ide.editorEmptyTitle} />
        )}
      </div>
    </section>
  )
}
