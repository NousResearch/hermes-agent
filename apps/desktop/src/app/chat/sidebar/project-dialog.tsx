import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'

import type { NewSessionPlacement } from '@/app/chat/new-session-drag'
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from '@/components/ui/select'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { isSubmitEnter } from '@/lib/ime'
import { type ProjectIdeaTemplate, randomIdeaTemplates } from '@/lib/project-idea-templates'
import { cn } from '@/lib/utils'
import type { ProjectInfo } from '@/types/hermes'
import { notifyError } from '@/store/notifications'
import { ALL_PROFILES, $profileScope, $profiles } from '@/store/profile'
import {
  $newProjectDropPlacement,
  $projectDialog,
  $projects,
  addProjectFolder,
  clearNewProjectDropPlacement,
  closeProjectDialog,
  createProject,
  enterProject,
  generateProjectIdea,
  pickProjectFolder,
  readProjectRow,
  removeProjectFolder,
  renameProject
} from '@/store/projects'

import { baseName } from './projects/workspace-groups'

// Single dialog mounted once in the sidebar; it renders create / rename /
// add-folder / remove-folder flows driven by the $projectDialog atom. Folders
// are chosen via the native directory picker (reused from the
// default-project-dir setting) — or, when removing one, picked from the list of
// folders the project already has.
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
  const scope = useStore($profileScope)
  const profiles = useStore($profiles)
  const profileNames = profiles.map(profile => profile.name)
  // The profile a create lands in (all-profiles view: the user picks it; any
  // other view: the write is ambient and this stays unused).
  const [targetProfile, setTargetProfile] = useState('')
  // Folder staged for removal (remove-folder mode only). Nothing is preselected:
  // the destructive button waits for an explicit pick.
  const [removePath, setRemovePath] = useState<null | string>(null)
  const nameRef = useRef<HTMLInputElement>(null)

  // A "New project" DRAG arms where the project should start (tab-strip slot /
  // pane edge / pane center) before the dialog opens. Snapshot it per open —
  // the submit forwards it as `dropPlacement`, and closing clears the store's
  // arm so a later plain-click create never inherits a stale placement.
  let dropPlacement: NewSessionPlacement | undefined

  if (open) {
    dropPlacement = $newProjectDropPlacement.get() ?? undefined
  }

  useEffect(() => {
    if (!open) {
      clearNewProjectDropPlacement()
    }
  }, [open])

  // Folders the target project still owns. The tree node carries none, so this
  // reads the cached project list — which every projects.* read-back folds
  // into. A merged all-profiles row is often owned by a profile the ambient
  // cache doesn't know; then the dialog fetches the picked profile's
  // authoritative list instead (remove-folder mode only).
  const cachedFolders =
    state?.projectId !== undefined ? (projects.find(proj => proj.id === state.projectId)?.folders ?? []) : []
  const [fetchedFolders, setFetchedFolders] = useState<ProjectInfo['folders'] | null>(null)

  useEffect(() => {
    setFetchedFolders(null)

    if (mode !== 'remove-folder' || !state?.projectId || cachedFolders.length > 0 || !state.profile) {
      return
    }

    let stale = false

    void readProjectRow(state.projectId, state.profile)
      .then(row => {
        if (!stale) {
          setFetchedFolders(row?.folders ?? [])
        }
      })
      .catch(() => {
        if (!stale) {
          setFetchedFolders([])
        }
      })

    return () => {
      stale = true
    }
  }, [mode, state?.projectId, state?.profile, cachedFolders.length])
  const projectFolders = fetchedFolders ?? cachedFolders

  useEffect(() => {
    if (open) {
      setName(state?.name ?? '')
      setFolders([])
      setIdea('')
      setTemplates(randomIdeaTemplates())
      setGeneratingIdea(false)
      setSubmitting(false)
      setRemovePath(null)
      setTargetProfile(
        state?.profile ??
          profiles.find(profile => profile.is_default)?.name ??
          profiles[0]?.name ??
          ''
      )

      if (mode !== 'add-folder' && mode !== 'remove-folder') {
        window.setTimeout(() => nameRef.current?.select(), 0)
      }
    }
  }, [open, mode, state?.name])

  const onOpenChange = (next: boolean) => {
    if (!next) {
      closeProjectDialog()
    }
  }

  // One submit beat for every flow: guard re-entry, run the write, close on
  // success, surface a toast on failure. Callers pass only the write, plus
  // optional hooks: onSuccess runs exactly when the write SUCCEEDS (before the
  // close — the New-project drop arm is consumed there, so a failed attempt
  // keeps its placement for the retry while a successful one can't leak it
  // forward), and failed names the failure headline when the default one would
  // misname the action.
  const runSubmit = async (
    write: () => Promise<unknown>,
    opts: { onSuccess?: () => void; failed?: string } = {}
  ) => {
    if (submitting) {
      return
    }

    setSubmitting(true)

    try {
      await write()
      opts.onSuccess?.()
      closeProjectDialog()
    } catch (err) {
      notifyError(err, opts.failed ?? p.createFailed)
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
        await runSubmit(() => addProjectFolder(projectId, dir, {}, state?.profile))

        return
      }

      setFolders(prev => (prev.includes(dir) ? prev : [...prev, dir]))

      // Picking a folder with no name typed names the project after the folder
      // (the ⌘O "Open folder…" naming), so one pick + Create is enough. The name
      // lands in the input, never in a hidden fallback the user cannot see.
      if (mode === 'create') {
        setName(prev => prev.trim() || baseName(dir) || prev)
      }
    } catch (err) {
      notifyError(err, p.createFailed)
    }
  }

  const submit = async () => {
    const trimmed = name.trim()
    const projectId = state?.projectId

    if (mode === 'rename' && projectId) {
      if (trimmed) {
        await runSubmit(() => renameProject(projectId, trimmed, state?.profile))
      }

      return
    }

    // The picker is the confirmation: the write only fires once a folder is
    // staged, and it lands in the profile the live session is on (same routing
    // as add-folder).
    if (mode === 'remove-folder' && projectId && removePath) {
      await runSubmit(() => removeProjectFolder(projectId, removePath, state?.profile), {
        failed: p.removeFolderFailed
      })

      return
    }

    // A project owns sessions by folder (cwd-prefix), so creation requires at
    // least one — a folder-less project couldn't hold a session anyway.
    if (mode === 'create' && trimmed && folders.length) {
      // The arm is consumed exactly on SUCCESS (before the close): a failed
      // create leaves the dialog open for a retry that still lands where it
      // was dropped; the open-state effect discards it on cancel/teardown.
      await runSubmit(async () => {
        const created = await createProject({
          dropPlacement,
          folders,
          idea: idea.trim() || undefined,
          name: trimmed,
          // In the all-profiles view the sidebar cannot name an owner, so the
          // selector's pick is the owner; elsewhere the write is ambient.
          profile: scope === ALL_PROFILES ? targetProfile : undefined,
          use: true
        })

        if (created) {
          enterProject(created.id)
        }
      }, { onSuccess: clearNewProjectDropPlacement })
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
    mode === 'create'
      ? p.createTitle
      : mode === 'rename'
        ? p.renameTitle
        : mode === 'add-folder'
          ? p.addFolderTitle
          : p.removeFolder

  const description = mode === 'create' ? p.createDesc : mode === 'remove-folder' ? p.removeFolderDesc : undefined

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-md" onInteractOutside={event => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle>{title}</DialogTitle>
          {description ? <DialogDescription>{description}</DialogDescription> : null}
        </DialogHeader>

        {mode !== 'add-folder' && mode !== 'remove-folder' && (
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setName(event.target.value)}
            onKeyDown={event => {
              if (isSubmitEnter(event)) {
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

        {mode === 'create' && scope === ALL_PROFILES && profileNames.length > 1 && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[0.6875rem] font-medium text-(--ui-text-tertiary)">{p.profileLabel}</span>
            <Select disabled={submitting} onValueChange={setTargetProfile} value={targetProfile || undefined}>
              <SelectTrigger className="w-full">
                <SelectValue placeholder={p.profileLabel} />
              </SelectTrigger>
              <SelectContent>
                {profileNames.map(name => (
                  <SelectItem key={name} value={name}>
                    {name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
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
                    <Tip label={p.removeFolder}>
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
                    </Tip>
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
              <Tip label={p.ideaShuffle}>
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
              </Tip>
            </div>
          </div>
        )}

        {mode === 'remove-folder' && (
          <div className="flex flex-col gap-1.5">
            <span className="text-[0.6875rem] font-medium text-(--ui-text-tertiary)">{p.foldersLabel}</span>
            {projectFolders.length === 0 ? (
              // Only reachable if the folders vanished (another window removed
              // them) between opening the menu and the dialog.
              <span className="text-[0.75rem] text-(--ui-text-quaternary)">{p.noFolders}</span>
            ) : (
              <ul className="flex flex-col gap-1">
                {projectFolders.map(folder => {
                  const selected = removePath === folder.path

                  return (
                    <li key={folder.path}>
                      <button
                        aria-pressed={selected}
                        className={cn(
                          'flex w-full items-center gap-2 rounded-md px-2 py-1 text-left text-[0.75rem] transition-colors disabled:opacity-50',
                          selected
                            ? 'bg-(--ui-control-active-background)'
                            : 'bg-(--ui-control-hover-background) hover:bg-(--ui-control-active-background)'
                        )}
                        disabled={submitting}
                        onClick={() => setRemovePath(folder.path)}
                        type="button"
                      >
                        <Codicon
                          className={cn('shrink-0', selected ? 'text-foreground' : 'text-transparent')}
                          name="check"
                          size="0.75rem"
                        />
                        <span className="min-w-0 flex-1 truncate" title={folder.path}>
                          {folder.label || baseName(folder.path) || folder.path}
                        </span>
                        {folder.is_primary && (
                          <span className="shrink-0 text-[0.625rem] uppercase text-(--ui-text-quaternary)">
                            {p.primaryBadge}
                          </span>
                        )}
                      </button>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        )}

        {mode === 'add-folder' && (
          <Button disabled={submitting} onClick={() => void pickFolder()} type="button">
            <Codicon name="folder-opened" size="0.875rem" />
            {p.addFolder}
          </Button>
        )}

        {mode !== 'add-folder' && (
          <DialogFooter>
            <Button disabled={submitting} onClick={() => onOpenChange(false)} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            {mode === 'remove-folder' ? (
              // Nothing is preselected, so the destructive button stays inert
              // until a folder is staged — the pick IS the confirmation.
              <Button
                disabled={submitting || !removePath}
                onClick={() => void submit()}
                type="button"
                variant="destructive"
              >
                {p.removeFolder}
              </Button>
            ) : (
              <Button
                disabled={submitting || !name.trim() || (mode === 'create' && folders.length === 0)}
                onClick={() => void submit()}
                type="button"
              >
                {mode === 'rename' ? t.common.save : p.create}
              </Button>
            )}
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  )
}
