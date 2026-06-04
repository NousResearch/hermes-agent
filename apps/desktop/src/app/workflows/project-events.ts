import type { WorkflowProject } from '@/types/workflow'

export const WORKFLOW_PROJECTS_CHANGED_EVENT = 'hermes:workflow-projects-changed'

export type WorkflowProjectChangeAction = 'archived' | 'created' | 'removed' | 'updated'

export interface WorkflowProjectChangeDetail {
  action: WorkflowProjectChangeAction
  project?: WorkflowProject
  projectId?: string
}

export function applyWorkflowProjectChange(
  projects: WorkflowProject[],
  detail: WorkflowProjectChangeDetail
): WorkflowProject[] {
  const projectId = detail.projectId ?? detail.project?.id

  if ((detail.action === 'archived' || detail.action === 'removed') && projectId) {
    return projects.filter(project => project.id !== projectId)
  }

  if (!detail.project || detail.project.archived) {
    return projects
  }

  const withoutProject = projects.filter(project => project.id !== detail.project?.id)
  return [detail.project, ...withoutProject]
}

export function dispatchWorkflowProjectsChanged(detail: WorkflowProjectChangeDetail): void {
  if (typeof window === 'undefined') {
    return
  }

  window.dispatchEvent(new CustomEvent(WORKFLOW_PROJECTS_CHANGED_EVENT, { detail }))
}

export function subscribeWorkflowProjectsChanged(
  handler: (detail: WorkflowProjectChangeDetail) => void
): () => void {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const listener = (event: Event) => {
    handler((event as CustomEvent<WorkflowProjectChangeDetail>).detail)
  }
  window.addEventListener(WORKFLOW_PROJECTS_CHANGED_EVENT, listener)

  return () => window.removeEventListener(WORKFLOW_PROJECTS_CHANGED_EVENT, listener)
}
