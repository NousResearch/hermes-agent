import type { Translations } from './types'

type ThreadCopy = Translations['assistant']['thread']

// The error card's layer fallbacks (lib/error-surface-copy.ts): used when a failure code has no own entry.
export const enErrorLayers: Pick<ThreadCopy, 'errorLayerBodies' | 'errorLayers'> = {
  errorLayers: {
    auth: 'Sign-in problem',
    billing: 'Out of credits',
    disk: 'Disk full',
    endpoint: "Can't reach your model server",
    gateway: "Couldn't send your message",
    generic: "Hermes couldn't finish this reply",
    provider: 'The AI service returned an error',
    runtime: 'Hermes hit a problem',
    streaming: 'The reply was cut off'
  },
  errorLayerBodies: {
    auth: 'The AI service rejected your sign-in. Check the credentials for this provider, then send your message again.',
    billing: 'Your account has no credits left for this provider. Top up or switch provider, then send again.',
    disk: 'Your disk is full, so Hermes could not save this conversation. Free some space, then retry.',
    endpoint: "Hermes can't reach your custom model server. Check that it is running, then send your message again.",
    gateway:
      'Hermes hit an internal problem starting this reply. Send your message again; if it keeps happening, send diagnostics.',
    generic: 'Something went wrong while Hermes was replying. Retry, or copy the details if it keeps happening.',
    provider: 'The AI service could not complete this request. Retry in a moment or switch provider.',
    runtime:
      'Hermes hit an internal problem starting this reply. Send your message again; if it keeps happening, send diagnostics.',
    streaming: 'The connection dropped before the reply finished. Retry to send it again.'
  }
}
