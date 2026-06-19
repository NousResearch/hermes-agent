import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { projectAgentsMdPath, readDesktopFileText, writeDesktopFileText } from '@/lib/desktop-fs'
import { cn } from '@/lib/utils'
import { $projects, getProject, updateProject } from '@/store/projects'

interface ProjectPageProps {
  projectId: string
  onStartSession?: (projectPath: string) => void
}

export function ProjectPageView({ projectId, onStartSession }: ProjectPageProps) {
  useStore($projects)
  const project = getProject(projectId)

  if (!project) {
    return (
      <div className="flex h-full w-full items-center justify-center text-sm text-muted-foreground">
        Project not found
      </div>
    )
  }

  return (
    <div className="flex h-full w-full min-w-0 overflow-hidden">
      <ProjectEmptyState projectTitle={project.title} />
      <ProjectSettingsPanel
        onStartSession={onStartSession}
        projectId={projectId}
      />
    </div>
  )
}

function ProjectEmptyState({ projectTitle }: { projectTitle: string }) {
  return (
    <div className="flex min-w-0 flex-1 flex-col items-center justify-center gap-3 text-muted-foreground">
      <Codicon name="folder-opened" size="2rem" />
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">{projectTitle}</p>
        <p className="mt-1 text-xs">
          Start a new session from the panel on the right to begin working in this project.
        </p>
      </div>
    </div>
  )
}

function ProjectSettingsPanel({
  onStartSession,
  projectId
}: {
  onStartSession?: (projectPath: string) => void
  projectId: string
}) {
  useStore($projects)
  const project = getProject(projectId)
  const [description, setDescription] = useState(project?.description ?? '')
  const [instructions, setInstructions] = useState('')
  const [instructionsLoaded, setInstructionsLoaded] = useState(false)
  const [instructionsSaving, setInstructionsSaving] = useState(false)

  const projectPath = project?.path ?? ''

  useEffect(() => {
    if (!project || !projectPath) { return }

    setInstructionsLoaded(false)
    readDesktopFileText(projectAgentsMdPath(projectPath))
      .then(result => {
        setInstructions(result.text)
        setInstructionsLoaded(true)
      })
      .catch(() => {
        setInstructions('')
        setInstructionsLoaded(true)
      })
  }, [projectPath, project])

  if (!project) {
    return null
  }

  const saveInstructions = async () => {
    if (!project || instructionsSaving) { return }

    setInstructionsSaving(true)

    try {
      await writeDesktopFileText(projectAgentsMdPath(projectPath), instructions)
    } catch {
      // silent
    } finally {
      setInstructionsSaving(false)
    }
  }

  const openFolder = () => {
    // Use hermesDesktop.openExternal with a file:// URL — works cross-platform
    // (Windows, macOS, Linux) via Electron's shell.openPath under the hood.
    const fileUrl = `file://${project.path.replaceAll('\\', '/')}`
    void window.hermesDesktop?.openExternal(fileUrl).catch(() => {})
  }

  const handleUpdateDescription = () => {
    updateProject(projectId, { description })
  }

  const folderName =
    project.path
      .split(/[\\/]+/)
      .filter(Boolean)
      .pop() ?? project.path

  return (
    <aside
      className={cn(
        'flex h-full flex-col overflow-hidden border-l border-(--ui-stroke-secondary) bg-(--ui-sidebar-surface-background) text-(--ui-text-tertiary)',
        'pt-(--titlebar-height)'
      )}
      style={{ maxWidth: 360, minWidth: 200, width: 280 }}
    >
      <div className="flex shrink-0 flex-col gap-0.5 border-b border-(--ui-stroke-secondary) px-3 py-3">
        <p className="truncate text-sm font-semibold text-foreground" title={project.title}>
          {project.title}
        </p>
        <p className="truncate text-[0.68rem] text-muted-foreground" title={project.path}>
          {folderName}
        </p>
      </div>

      <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-3 py-3">
        <div className="grid gap-1.5">
          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-foreground">Instructions</label>
            <Button
              className="h-5 px-1.5 text-[0.6875rem]"
              disabled={!instructionsLoaded || instructionsSaving}
              onClick={() => void saveInstructions()}
              size="sm"
              variant="ghost"
            >
              {instructionsSaving ? 'Saving...' : 'Save'}
            </Button>
          </div>
          <p className="text-[0.68rem] text-muted-foreground">Saved as AGENTS.md in the project folder</p>
          <Textarea
            className="min-h-28 resize-none font-mono text-xs leading-5"
            disabled={!instructionsLoaded}
            onBlur={() => void saveInstructions()}
            onChange={e => setInstructions(e.target.value)}
            placeholder="Add instructions for this project... Saved as AGENTS.md in the project folder."
            value={instructions}
          />
        </div>

        <div className="grid gap-1.5">
          <label className="text-xs font-medium text-foreground">Description</label>
          <div className="flex gap-1.5">
            <Input
              className="h-7 text-xs"
              maxLength={40}
              onChange={event => setDescription(event.target.value)}
              placeholder="Short description"
              value={description}
            />
            <Button className="h-7 shrink-0 text-xs" onClick={handleUpdateDescription} size="sm" variant="outline">
              Update
            </Button>
          </div>
        </div>
      </div>

      <div className="shrink-0 border-t border-(--ui-stroke-secondary) px-3 py-3">
        <Button
          className="w-full justify-start gap-1.5 text-xs"
          onClick={() => onStartSession?.(project.path)}
          size="sm"
          variant="outline"
        >
          <Codicon name="play" size="0.875rem" />
          Start A New Session
        </Button>
      </div>

      <div className="shrink-0 border-t border-(--ui-stroke-secondary) px-3 py-3">
        <Button className="w-full justify-start gap-1.5 text-xs" onClick={openFolder} size="sm" variant="outline">
          <Codicon name="folder-opened" size="0.875rem" />
          Open in Explorer
        </Button>
      </div>
    </aside>
  )
}
