import type { QueueFairCopy } from './types_queue_fair'

export const enQueueFair: QueueFairCopy = {
  holdUnknown: reason => `Waiting: ${reason}`,
  hold: {
    session_busy: 'Waiting for the current turn',
    slot: 'Waiting for a free session slot',
    editing: 'Being edited, will not start yet',
    restarting: 'Waiting for the gateway to restart',
    paused: 'Paused until the lost turn is resolved',
    provider_limit: 'Waiting for the provider limit to reset',
    memory: 'Waiting for free memory'
  },
  moveUp: 'Move up',
  moveDown: 'Move down',
  holdFailed: 'Could not pause this queued message while you edit it; it may start before you save'
}
