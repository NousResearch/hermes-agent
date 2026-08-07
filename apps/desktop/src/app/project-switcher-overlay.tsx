import { useStore } from '@nanostores/react'
import { useCallback } from 'react'

import { ProjectSwitcherDialog } from '@/components/project-switcher'
import { $projectSwitcherOpen, setProjectSwitcherOpen } from '@/store/project-switcher'
import { goToProject, openFolderAsProject } from '@/store/projects'

/**
 * Presentation shell for the focused project picker. Project identity and
 * navigation stay in the shared Projects model; this overlay owns no paths or
 * durable state of its own.
 */
export function ProjectSwitcherOverlay() {
  const open = useStore($projectSwitcherOpen)
  const handleOpenFolder = useCallback(() => void openFolderAsProject(), [])
  const handleSelect = useCallback((projectId: string) => goToProject(projectId, { newSession: true }), [])

  return (
    <ProjectSwitcherDialog
      onOpenChange={setProjectSwitcherOpen}
      onOpenFolder={handleOpenFolder}
      onSelect={handleSelect}
      open={open}
    />
  )
}
