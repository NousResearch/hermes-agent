import { atom } from 'nanostores'

export const $workflowCopilotOpen = atom(false)
export const $workflowCopilotExpanded = atom(false)

export function setWorkflowCopilotOpen(open: boolean) {
  $workflowCopilotOpen.set(open)
}

export function toggleWorkflowCopilotOpen() {
  $workflowCopilotOpen.set(!$workflowCopilotOpen.get())
}

export function setWorkflowCopilotExpanded(expanded: boolean) {
  $workflowCopilotExpanded.set(expanded)
}

export function toggleWorkflowCopilotExpanded() {
  $workflowCopilotExpanded.set(!$workflowCopilotExpanded.get())
}
