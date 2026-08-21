export type WindowCloseAction = 'cancel' | 'minimize' | 'quit'
export type WindowCloseDecision = 'hold' | 'proceed' | 'prompt'

interface WindowClosePromptInput {
  handoffInProgress?: boolean
  platform: NodeJS.Platform
  promptOpen?: boolean
  quitInProgress?: boolean
}

export function windowCloseDecision({
  handoffInProgress = false,
  platform,
  promptOpen = false,
  quitInProgress = false
}: WindowClosePromptInput): WindowCloseDecision {
  if (platform === 'darwin' || handoffInProgress || quitInProgress) {
    return 'proceed'
  }

  return promptOpen ? 'hold' : 'prompt'
}

export function closeActionForResponse(response: number): WindowCloseAction {
  if (response === 0) {
    return 'minimize'
  }

  if (response === 1) {
    return 'quit'
  }

  return 'cancel'
}
