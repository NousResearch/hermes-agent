import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useId, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { cn } from '@/lib/utils'
import { notify } from '@/store/notifications'
import {
  $portalCreateProjectDialog,
  closePortalCreateProjectDialog,
  createPortalProject,
  deletePortalProject,
  PORTAL_PROJECT_NAME_MAX
} from '@/store/portal-projects'

/**
 * "Create a new project" modal (Figma 1713:101964, 1713:102008, 1715:9972).
 * Flow A (opened from a chat's Move to project submenu): creates the project and moves that chat.
 * Flow B (opened from an agent's menu): creates an empty project under that agent.
 * Both show a toast with Undo that deletes the project (and returns the chat to loose chats).
 */
export function PortalCreateProjectDialog() {
  const request = useStore($portalCreateProjectDialog)
  const [name, setName] = useState('')
  const inputRef = useRef<HTMLInputElement | null>(null)
  const inputId = useId()
  const errorId = useId()

  useEffect(() => {
    if (request) {
      setName('')
      requestAnimationFrame(() => inputRef.current?.focus())
    }
  }, [request])

  const trimmed = name.trim()
  const tooLong = name.length > PORTAL_PROJECT_NAME_MAX
  const canSubmit = trimmed.length > 0 && !tooLong

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()

    if (!request || !canSubmit) {
      return
    }

    const sessionId = request.session?.id
    const created = createPortalProject(request.owner, trimmed, sessionId)

    if (!created) {
      return
    }

    const { owner } = request
    closePortalCreateProjectDialog()

    notify({
      action: {
        label: 'Undo',
        onClick: () => {
          deletePortalProject(
            owner,
            created.project.id,
            sessionId ? { previousProjectId: created.previousProjectId, sessionId } : undefined
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
      <DialogContent className="portal-create-project-dialog sm:max-w-[25.375rem]" data-slot="portal-create-project-dialog">
        {request ? (
          <form className="portal-create-project-form" noValidate onSubmit={handleSubmit}>
            <DialogHeader className="portal-create-project-header">
              <DialogTitle className="portal-create-project-title">Create a new project</DialogTitle>
              <DialogDescription className="portal-create-project-body">
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
                aria-describedby={tooLong ? errorId : undefined}
                aria-invalid={tooLong || undefined}
                autoComplete="off"
                autoFocus
                className={cn('portal-create-project-input', tooLong && 'is-error')}
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
              ) : null}
            </div>

            <DialogFooter className="portal-create-project-footer">
              <Button onClick={closePortalCreateProjectDialog} type="button" variant="ghost">
                Cancel
              </Button>
              <Button className="portal-create-project-submit" type="submit">
                {request.session ? 'Create and move' : 'Create project'}
              </Button>
            </DialogFooter>
          </form>
        ) : null}
      </DialogContent>
    </Dialog>
  )
}
