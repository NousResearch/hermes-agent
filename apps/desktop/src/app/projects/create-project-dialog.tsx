import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { projectAgentsMdPath, readDesktopFileText, selectDesktopPaths, writeDesktopFileText } from '@/lib/desktop-fs'
import { AlertTriangle } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { addProject, updateProject } from '@/store/projects'
import type { Project } from '@/store/projects'

export function CreateProjectDialog({
  onClose,
  onCreated,
  open,
  project
}: {
  onClose: () => void
  onCreated: (project: Project) => void
  open: boolean
  project?: Project
}) {
  const editMode = !!project

  const [title, setTitle] = useState('')
  const [description, setDescription] = useState('')
  const [path, setPath] = useState('')
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setTitle(project?.title ?? '')
    setDescription(project?.description ?? '')
    setPath(project?.path ?? '')
    setError(null)
    setStatus('idle')
  }, [open, project])

  const busy = status === 'saving' || status === 'done'
  const trimmedTitle = title.trim()
  const trimmedDescription = description.trim()

  const chooseFolder = async () => {
    const selected = await selectDesktopPaths({ directories: true, multiple: false })

    if (selected?.[0]) {
      setPath(selected[0])
    }
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (!editMode && !path) {
      setError('Please choose a folder for the project.')

      return
    }

    if (!trimmedTitle) {
      setError('Project name is required.')

      return
    }

    setStatus('saving')
    setError(null)

    try {
      if (editMode) {
        updateProject(project.id, { title: trimmedTitle, description: trimmedDescription })
        setStatus('done')
        window.setTimeout(onClose, 400)
      } else {
        // Auto-create AGENTS.md only if it doesn't already exist in the folder.
        const agentsMdPath = projectAgentsMdPath(path)

        const alreadyExists = await readDesktopFileText(agentsMdPath)
          .then(() => true)
          .catch(() => false)

        if (!alreadyExists) {
          const starterContent = [
            `# ${trimmedTitle}`,
            trimmedDescription ? `\n${trimmedDescription}` : '',
            '\n\n<!-- Add project-specific instructions for the AI agent here. -->'
          ].join('')

          await writeDesktopFileText(agentsMdPath, starterContent)
        }

        const created = addProject({ title: trimmedTitle, description: trimmedDescription, path })
        onCreated(created)
        setStatus('done')
        window.setTimeout(onClose, 400)
      }
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : editMode ? 'Failed to save project.' : 'Failed to create project.')
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{editMode ? 'Edit Project' : 'New Project'}</DialogTitle>
          <DialogDescription>
            {editMode
              ? 'Update the project name and description.'
              : 'Choose a folder for your project. An AGENTS.md file in that folder will be used as project instructions.'}
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          {!editMode && (
            <div className="grid gap-1.5">
              <Button onClick={() => void chooseFolder()} type="button" variant="outline">
                Choose Folder
              </Button>
              <p className={cn('truncate text-xs', path ? 'text-foreground' : 'text-muted-foreground')}>
                {path || 'No folder selected'}
              </p>
            </div>
          )}

          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="new-project-title">
              Project Name
            </label>
            <Input
              autoFocus
              id="new-project-title"
              onChange={event => setTitle(event.target.value)}
              placeholder="My Project"
              value={title}
            />
          </div>

          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="new-project-description">
              Description <span className="font-normal text-muted-foreground">— optional</span>
            </label>
            <Input
              id="new-project-description"
              maxLength={40}
              onChange={event => setDescription(event.target.value)}
              placeholder="Short description of this project"
              value={description}
            />
            <p className="text-[0.66rem] text-muted-foreground">{description.length}/40</p>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={busy || !trimmedTitle || (!editMode && !path)} type="submit">
              {editMode
                ? status === 'saving' ? 'Saving…' : status === 'done' ? 'Saved' : 'Save'
                : status === 'saving' ? 'Creating…' : status === 'done' ? 'Created' : 'Create'}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
