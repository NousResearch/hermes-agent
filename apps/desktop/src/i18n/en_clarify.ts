import type { ClarifyCopy } from './types_clarify'

export const enClarify: ClarifyCopy = {
  notReady: 'Clarify request is not ready yet',
  gatewayDisconnected: 'Hermes is offline right now. Reconnect, then send it again.',
  sendFailed: 'Could not send clarify response',
  loadingQuestion: 'Loading question…',
  other: 'Other (type your answer)',
  placeholder: 'Type your answer…',
  skip: 'Skip',
  skipped: 'Skipped',
  noAnswer: 'No answer',
  confirmAndContinueLabel: 'Confirm and continue',
  singleSelectHint: 'Pick one',
  multiSelectHint: 'Select all that apply',
  oneQuestion: '1 question',
  questionProgress: (answered, total) => `${answered} of ${total} answered`,
  notDelivered:
    "This question didn't reach the app, so it can't be answered here. Press Stop to end the turn, then reply in chat."
}
