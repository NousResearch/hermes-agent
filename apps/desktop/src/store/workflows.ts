import { atom } from 'nanostores'

export type WorkflowPhaseStatus = 'completed' | 'failed' | 'pending' | 'running' | 'skipped'

export interface WorkflowPhaseProgress {
  id: string
  status: WorkflowPhaseStatus
  title: string
}

export interface WorkflowProgress {
  id: string
  phases: WorkflowPhaseProgress[]
  title: string
}

export const $workflowProgressBySession = atom<Record<string, WorkflowProgress>>({})

export function setSessionWorkflowProgress(sid: string, progress: WorkflowProgress | null) {
  if (!sid) {
    return
  }

  const current = $workflowProgressBySession.get()

  if (!progress) {
    if (!(sid in current)) {
      return
    }

    const { [sid]: _drop, ...rest } = current
    $workflowProgressBySession.set(rest)

    return
  }

  $workflowProgressBySession.set({ ...current, [sid]: progress })
}
