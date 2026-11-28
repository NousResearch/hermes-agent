import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { PathText } from '@/components/ui/path-text'
import { cn } from '@/lib/utils'
import { notify } from '@/store/notifications'
import {
  $portalAgentProjects,
  $portalCreateProjectDialog,
  agentPortalProjectsFor,
  closePortalCreateProjectDialog,
  PORTAL_PROJECT_NAME_MAX
} from '@/store/portal-projects'
import { createPortalProjectSynced, deletePortalProjectSynced } from '@/store/portal-projects-sync'
import { pickProjectFolder } from '@/store/projects'
/**
 * "Create a new project" modal (Figma 1713:101964, 1713:102008, 1715:9972).
 * Flow A (opened from a chat's Move to project submenu): creates the project and moves that chat.
 * Flow B (opened from an agent's menu): creates an empty project under that agent.
 * Both show a toast with Undo that deletes the project (and returns the chat to loose chats).
 */
export function PortalCreateProjectDialog() {
  const request = useStore($portalCreateProjectDialog)
  const allProjects = useStore($portalAgentProjects)
  const [name, setName] = useState('')
  const [folder, setFolder] = useState<string | null>(null)
  const [pickingFolder, setPickingFolder] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const inputId = useId()
  const errorId = useId()

  useEffect(() => {
    if (request) {
      setName('')
      setFolder(null)
      setPickingFolder(false)
      setSubmitting(false)
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [request])

  const trimmed = name.trim()
  const tooLong = name.length > PORTAL_PROJECT_NAME_MAX
  const duplicateName =
    Boolean(request && trimmed) &&
    agentPortalProjectsFor(allProjects, request!.owner).projects.some(
      project => project.name.trim().toLowerCase() === trimmed.toLowerCase()
    )
  const hasError = tooLong || duplicateName
  const canSubmit = trimmed.length > 0 && !hasError && !submitting

  const handlePickFolder = async () => {
    if (pickingFolder || !request) {
      return
    }

    const activeRequest = request
    setPickingFolder(true)

    try {
      const picked = await pickProjectFolder()

      if (picked && $portalCreateProjectDialog.get() === activeRequest) {
        setFolder(picked)
      }
    } catch {
      // User cancelled or picker unavailable.
    } finally {
      if ($portalCreateProjectDialog.get() === activeRequest) {
        setPickingFolder(false)
      }
    }
  }

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()

    if (!request || !canSubmit) {
      return
    }

    const activeRequest = request
    setSubmitting(true)
    const sessionId = activeRequest.session?.id
    const { owner } = activeRequest
    const currentEntry = agentPortalProjectsFor($portalAgentProjects.get(), owner)
    const previousProject = sessionId
      ? currentEntry.projects.find(project => project.id === currentEntry.sessionProject[sessionId])
      : undefined
    const created = await createPortalProjectSynced(owner, trimmed, {
      folder,
      moveSessionId: sessionId
    })

    if ($portalCreateProjectDialog.get() !== activeRequest) {
      return
    }

    setSubmitting(false)
    if (!created) {
      return
    }

    closePortalCreateProjectDialog()

    notify({
      action: {
        label: 'Undo',
        onClick: () => {
          deletePortalProjectSynced(
            owner,
            created.project,
            sessionId
              ? {
                  previousProjectId: created.previousProjectId,
                  previousProjectPath: previousProject?.path ?? null,
                  sessionId
                }
              : undefined
          )
        }
      },
      durationMs: 5_000,
      kind: 'success',
      message: `${created.project.name} created.`
    })
  }

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          closePortalCreateProjectDialog()
        }
      }}
      open={request !== null}
    >
      <DialogContent className="sm:max-w-md" data-slot="portal-create-project-dialog">
        {request ? (
          <form className="portal-create-project-form" noValidate onSubmit={handleSubmit}>
            <DialogHeader>
              <DialogTitle>Create a new project</DialogTitle>
              <DialogDescription>
                {request.session
                  ? `Create a project for ${request.agentLabel} and move ${request.session.title}.`
                  : `Create a project for ${request.agentLabel}.`}
              </DialogDescription>
            </DialogHeader>

            <div className="portal-create-project-field">
              <div className="portal-create-project-label-row">
                <label className="portal-create-project-label" htmlFor={inputId}>
                  Name
                </label>
                <span
                  aria-live="polite"
                  className={cn('portal-create-project-counter', tooLong && 'is-error')}
                  data-slot="project-name-counter"
                >
                  {name.length}/{PORTAL_PROJECT_NAME_MAX}
                </span>
              </div>
              <Input
                aria-describedby={hasError ? errorId : undefined}
                aria-invalid={hasError || undefined}
                autoComplete="off"
                autoFocus
                className={cn(hasError && 'portal-create-project-input is-error')}
                id={inputId}
                onChange={event => setName(event.target.value)}
                placeholder="Launch prep"
                ref={inputRef}
                value={name}
              />
              {tooLong ? (
                <p className="portal-create-project-error" id={errorId} role="alert">
                  Use {PORTAL_PROJECT_NAME_MAX} characters or fewer so the name fits in the sidebar.
                </p>
              ) : duplicateName ? (
                <p className="portal-create-project-error" id={errorId} role="alert">
                  A project named &ldquo;{trimmed}&rdquo; already exists for this agent.
                </p>
              ) : null}
            </div>

            <div className="portal-create-project-folder" data-slot="portal-create-project-folder">
              <div className="portal-create-project-label-row">
                <span className="portal-create-project-label">Folder</span>
                <span className="portal-create-project-counter">Optional</span>
              </div>
              {folder ? (
                <div className="portal-create-project-folder-chip">
                  <Codicon aria-hidden name="folder-opened" size="0.875rem" />
                  <PathText className="min-w-0 flex-1 truncate" path={folder} />
                  <Button
                    aria-label="Remove folder"
                    onClick={() => setFolder(null)}
                    size="icon-xs"
                    type="button"
                    variant="ghost"
                  >
                    <Codicon aria-hidden name="close" size="0.75rem" />
                  </Button>
                </div>
              ) : (
                <Button
                  className="justify-start"
                  disabled={pickingFolder}
                  onClick={() => void handlePickFolder()}
                  size="sm"
                  type="button"
                  variant="outline"
                >
                  <Codicon aria-hidden name="folder-opened" size="0.875rem" />
                  {pickingFolder ? 'Choosing folder…' : 'Attach folder…'}
                </Button>
              )}
              <p className="portal-create-project-hint">
                New chats in this project start in this folder with Git and file tools attached.
              </p>
            </div>

            <DialogFooter>
              <Button onClick={closePortalCreateProjectDialog} type="button" variant="outline">
                Cancel
              </Button>
              <Button disabled={!canSubmit} type="submit">
                {submitting ? 'Creating…' : request.session ? 'Create and move' : 'Create project'}
              </Button>
            </DialogFooter>
          </form>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
