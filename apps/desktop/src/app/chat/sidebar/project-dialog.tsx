import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

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
import { GenerateButton } from '@/components/ui/generate-button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { useI18n } from '@/i18n'
import { type ProjectIdeaTemplate, randomIdeaTemplates } from '@/lib/project-idea-templates'
import { cn } from '@/lib/utils'
import { notifyError } from '@/store/notifications'
import {
  $projects,
  $projectDialog,
  addProjectFolder,
  closeProjectDialog,
  createProject,
  generateProjectIdea,
  moveProjectFolder,
  pickProjectFolder,
  renameProject
} from '@/store/projects'

// Local-only edit state for the edit-folders dialog. Kept outside the atom so
// the user can stage several folder moves before committing them in one shot —
// the per-row optimistic cache update only fires on Save.
interface FolderEdit {
  original: string
  current: string
}

// Single dialog mounted once in the sidebar; it renders create / rename /
// add-folder / edit-folders flows driven by the $projectDialog atom. Folders
// are chosen via the native directory picker (reused from the default-project-
// dir setting).
export function ProjectDialog() {
  const { t } = useI18n()
  const p = t.sidebar.projects
  const state = useStore($projectDialog)
  const projects = useStore($projects)
  const open = state !== null
  const mode = state?.mode ?? 'create'

  const [name, setName] = useState('')
  const [folders, setFolders] = useState<string[]>([])
  const [idea, setIdea] = useState('')
  const [templates, setTemplates] = useState<ProjectIdeaTemplate[]>([])
  const [generatingIdea, setGeneratingIdea] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [folderEdits, setFolderEdits] = useState<FolderEdit[]>([])
  const nameRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (open) {
      setName(state?.name ?? '')
      setFolders([])
      setIdea('')
      setTemplates(randomIdeaTemplates())
      setGeneratingIdea(false)
      setSubmitting(false)

      // Seed the edit-folders staging buffer from the project's current
      // folders. Done here (not in render) so a re-open resets in-flight edits
      // and so a concurrent refresh can't sneak a fresh folder in mid-edit.
      if (mode === 'edit-folders') {
        const proj = projects.find(p2 => p2.id === state?.projectId)
        setFolderEdits((proj?.folders ?? []).map(f => ({ original: f.path, current: f.path })))
      } else {
        setFolderEdits([])
      }

      if (mode !== 'add-folder') {
        window.setTimeout(() => nameRef.current?.select(), 0)
      }
    }
  }, [open, mode, state?.name, state?.projectId, projects])

  const onOpenChange = (next: boolean) => {
    if (!next) {
      closeProjectDialog()
    }
  }

  // One submit beat for every flow: guard re-entry, run the write, close on
  // success, surface a toast on failure. Callers pass only the write.
  const runSubmit = async (write: () => Promise<unknown>) => {
    if (submitting) {
      return
    }

    setSubmitting(true)

    try {
      await write()
      closeProjectDialog()
    } catch (err) {
      notifyError(err, p.createFailed)
    } finally {
      setSubmitting(false)
    }
  }

  const pickFolder = async () => {
    try {
      const dir = await pickProjectFolder()

      if (!dir) {
        return
      }

      const projectId = state?.projectId

      if (mode === 'add-folder' && projectId) {
        await runSubmit(() => addProjectFolder(projectId, dir))

        return
      }

      setFolders(prev => (prev.includes(dir) ? prev : [...prev, dir]))
    } catch (err) {
      notifyError(err, p.createFailed)
    }
  }

  // Pick a new path for one row in the edit-folders staging buffer. The picker
  // returns an absolute path; we replace just that row's `current` so the user
  // can stage several moves before committing them all in one submit.
  const pickFolderEdit = async (originalPath: string) => {
    try {
      const dir = await pickProjectFolder()

      if (!dir) {
        return
      }

      setFolderEdits(prev =>
        prev.map(f => (f.original === originalPath ? { ...f, current: dir } : f))
      )
    } catch (err) {
      notifyError(err, p.editFoldersMoveFailed)
    }
  }

  const submit = async () => {
    const trimmed = name.trim()
    const projectId = state?.projectId

    if (mode === 'rename' && projectId) {
      if (trimmed) {
        await runSubmit(() => renameProject(projectId, trimmed))
      }

      return
    }

    // A project owns sessions by folder (cwd-prefix), so creation requires at
    // least one — a folder-less project couldn't hold a session anyway.
    if (mode === 'create' && trimmed && folders.length) {
      await runSubmit(() =>
        createProject({ folders, idea: idea.trim() || undefined, name: trimmed, use: true })
      )
      return
    }

    // Commit staged folder moves in original-path order. We process them
    // sequentially so the cache stays consistent: a later move's optimistic
    // write sees the prior move's already-updated primary_path / repos.
    // Backend `projects.move_folder` is idempotent for same-path, so an empty
    // edit closes immediately.
    if (mode === 'edit-folders' && projectId) {
      const moves = folderEdits.filter(f => f.original !== f.current)
      if (!moves.length) {
        closeProjectDialog()
        return
      }

      await runSubmit(async () => {
        for (const move of moves) {
          await moveProjectFolder(projectId, move.original, move.current)
        }
      })
    }
  }

  const generateIdea = async () => {
    if (generatingIdea) {
      return
    }

    setGeneratingIdea(true)

    try {
      const text = await generateProjectIdea(name)

      if (text) {
        setIdea(text)
      }
    } finally {
      setGeneratingIdea(false)
    }
  }

  const title =
    mode === 'rename'
      ? p.renameTitle
      : mode === 'add-folder'
        ? p.addFolderTitle
        : mode === 'edit-folders'
          ? p.editFoldersTitle
          : p.createTitle

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md" onInteractOutside={event => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {mode === 'create' && <DialogDescription>{p.createDesc}</DialogDescription>}
          {mode === 'edit-folders' && <DialogDescription>{p.editFoldersDesc}</DialogDescription>}
        </DialogHeader>

        {mode !== 'add-folder' && mode !== 'edit-folders' && (
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setName(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Enter') {
                event.preventDefault()
                void submit()
              } else if (event.key === 'Escape') {
                onOpenChange(false)
              }
            }}
            placeholder={p.namePlaceholder}
            ref={nameRef}
            value={name}
          />
        )}

        {mode === 'create' && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[0.6875rem] font-medium text-(--ui-text-tertiary)">{p.foldersLabel}</span>
            {folders.length === 0 ? (
              <span className="text-[0.75rem] text-(--ui-text-quaternary)">{p.noFolders}</span>
            ) : (
              <ul className="flex flex-col gap-1">
                {folders.map((folder, index) => (
                  <li
                    className={cn(
                      'flex items-center gap-2 rounded-md bg-(--ui-control-hover-background) px-2 py-1 text-[0.75rem]'
                    )}
                    key={folder}
                  >
                    <Codicon className="shrink-0 text-(--ui-text-tertiary)" name="folder" size="0.75rem" />
                    <span className="min-w-0 flex-1 truncate" title={folder}>
                      {folder}
                    </span>
                    {index === 0 && (
                      <span className="shrink-0 text-[0.625rem] uppercase text-(--ui-text-quaternary)">
                        {p.primaryBadge}
                      </span>
                    )}
                    <Button
                      aria-label={p.removeFolder}
                      className="size-5 shrink-0 text-(--ui-text-quaternary) hover:text-foreground"
                      onClick={() => setFolders(prev => prev.filter(f => f !== folder))}
                      size="icon-xs"
                      type="button"
                      variant="ghost"
                    >
                      <Codicon name="close" size="0.75rem" />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
            <Button
              className="self-start"
              disabled={submitting}
              onClick={() => void pickFolder()}
              size="sm"
              type="button"
              variant="ghost"
            >
              <Codicon name="add" size="0.75rem" />
              {p.addFolder}
            </Button>
          </div>
        )}

        {mode === 'create' && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[0.6875rem] font-medium text-(--ui-text-tertiary)">{p.ideaLabel}</span>
            <div className="relative">
              <Textarea
                className="min-h-20 pr-8 text-[0.8125rem]"
                disabled={submitting}
                onChange={event => setIdea(event.target.value)}
                placeholder={p.ideaPlaceholder}
                value={idea}
              />
              <GenerateButton
                className="absolute top-1 right-1"
                disabled={submitting}
                generating={generatingIdea}
                generatingLabel={p.ideaGenerating}
                label={p.ideaGenerate}
                onGenerate={() => void generateIdea()}
              />
            </div>
            <div className="flex flex-wrap items-center gap-1">
              {templates.map(template => (
                <button
                  className="flex items-center gap-1 rounded-full border border-(--ui-stroke-tertiary) px-2 py-0.5 text-[0.6875rem] text-(--ui-text-secondary) transition-colors hover:border-(--ui-stroke-secondary) hover:bg-(--ui-control-hover-background) hover:text-foreground disabled:opacity-50"
                  disabled={submitting}
                  key={template.label}
                  onClick={() => setIdea(template.idea)}
                  type="button"
                >
                  <span aria-hidden>{template.emoji}</span>
                  {template.label}
                </button>
              ))}
              <Button
                aria-label={p.ideaShuffle}
                className="size-5 text-(--ui-text-quaternary) hover:text-foreground"
                disabled={submitting}
                onClick={() => setTemplates(randomIdeaTemplates())}
                size="icon-xs"
                type="button"
                variant="ghost"
              >
                <Codicon name="refresh" size="0.75rem" />
              </Button>
            </div>
          </div>
        )}

        {mode === 'add-folder' && (
          <Button disabled={submitting} onClick={() => void pickFolder()} type="button">
            <Codicon name="folder-opened" size="0.875rem" />
            {p.addFolder}
          </Button>
        )}

        {mode === 'edit-folders' && (
          <div className="flex flex-col gap-2">
            <span className="text-[0.6875rem] font-medium text-(--ui-text-tertiary)">{p.foldersLabel}</span>
            {folderEdits.length === 0 ? (
              <span className="text-[0.75rem] text-(--ui-text-quaternary)">{p.editFoldersEmpty}</span>
            ) : (
              <ul className="flex flex-col gap-1.5">
                {folderEdits.map(edit => {
                  const changed = edit.original !== edit.current
                  // Collision: a staged `current` matches another row's `original`
                  // (the only case the backend can reject), or duplicates another
                  // row's staged `current` (would only happen across two rows
                  // pointing at the same new path).
                  const otherOriginals = new Set(
                    folderEdits.filter(o => o.original !== edit.original).map(o => o.original)
                  )
                  const otherCurrents = folderEdits
                    .filter(o => o.original !== edit.original)
                    .map(o => o.current)
                  const collision =
                    otherCurrents.includes(edit.current) ||
                    (changed && otherOriginals.has(edit.current))
                  return (
                    <li
                      className={cn(
                        'flex items-center gap-2 rounded-md bg-(--ui-control-hover-background) px-2 py-1 text-[0.75rem]'
                      )}
                      key={edit.original}
                    >
                      <Codicon
                        className="shrink-0 text-(--ui-text-tertiary)"
                        name={changed ? 'arrow-right' : 'folder'}
                        size="0.75rem"
                      />
                      <div className="flex min-w-0 flex-1 flex-col">
                        <span
                          className={cn(
                            'truncate text-(--ui-text-quaternary)',
                            changed && 'line-through'
                          )}
                          title={edit.original}
                        >
                          {edit.original}
                        </span>
                        {changed && (
                          <span
                            className={cn(
                              'truncate',
                              collision ? 'text-(--ui-text-danger, #f48771)' : 'text-foreground'
                            )}
                            title={edit.current}
                          >
                            {edit.current}
                          </span>
                        )}
                      </div>
                      {changed && (
                        <span className="shrink-0 text-[0.625rem] uppercase text-(--ui-text-quaternary)">
                          {p.editFolderChanged}
                        </span>
                      )}
                      <Button
                        aria-label={p.editFolderPick}
                        className="size-5 shrink-0 text-(--ui-text-quaternary) hover:text-foreground"
                        disabled={submitting}
                        onClick={() => void pickFolderEdit(edit.original)}
                        size="icon-xs"
                        type="button"
                        variant="ghost"
                      >
                        <Codicon name="folder-opened" size="0.75rem" />
                      </Button>
                      {changed && (
                        <Button
                          aria-label={p.editFolderRevert}
                          className="size-5 shrink-0 text-(--ui-text-quaternary) hover:text-foreground"
                          disabled={submitting}
                          onClick={() =>
                            setFolderEdits(prev =>
                              prev.map(f =>
                                f.original === edit.original ? { ...f, current: f.original } : f
                              )
                            )
                          }
                          size="icon-xs"
                          type="button"
                          variant="ghost"
                        >
                          <Codicon name="discard" size="0.75rem" />
                        </Button>
                      )}
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        )}

        {(mode === 'add-folder' || mode === 'edit-folders') && (
          <DialogFooter>
            <Button
              disabled={submitting}
              onClick={() => onOpenChange(false)}
              type="button"
              variant="ghost"
            >
              {t.common.cancel}
            </Button>
            <Button
              disabled={
                submitting ||
                // Block Save when no edits were staged — submit() would close
                // immediately, but a stray Enter shouldn't get a no-op reply.
                (mode === 'edit-folders' && folderEdits.every(f => f.original === f.current))
              }
              onClick={() => void submit()}
              type="button"
            >
              {mode === 'edit-folders' ? p.editFoldersDone : p.addFolder}
            </Button>
          </DialogFooter>
        )}

        {mode !== 'add-folder' && mode !== 'edit-folders' && (
          <DialogFooter>
            <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button
              disabled={submitting || !name.trim() || (mode === 'create' && folders.length === 0)}
              onClick={() => void submit()}
              type="button"
            >
              {mode === 'rename' ? t.common.save : p.create}
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}