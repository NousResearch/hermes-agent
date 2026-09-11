import type { ClientSessionState } from '@/app/types'
import { textPart } from '@/lib/chat-messages'
import type { BusyInputMode } from '@/store/busy-input-mode'
import type { ComposerAttachment } from '@/store/composer'
import { enqueueQueuedPrompt } from '@/store/composer-queue'
import { notify } from '@/store/notifications'
import { $sessions, resolveComposerSessionKey } from '@/store/session'
import { $sessionStates } from '@/store/session-states'

import { queueKickoffIfSessionBusy } from './queue-if-busy'
import { isTargetSessionBusy, slashStatusText } from './utils'
import type { SubmitTextOptions } from './utils'

export interface SteeringOptions {
  request: (method: string, params: Record<string, unknown>) => Promise<unknown>
  sessionId: string
  text: string
  render: (output: string) => void
  send: (text: string) => Promise<boolean> | boolean
}

export async function dispatchSteering({ request, sessionId, text, render, send }: SteeringOptions): Promise<boolean> {
  if (!sessionId || !text.trim()) {
    return false
  }

  const result = await request('command.dispatch', { session_id: sessionId, name: 'steer', arg: text })

  if (!result || typeof result !== 'object') {
    return false
  }

  const dispatch = result as { type?: string; output?: unknown; message?: unknown }

  if (dispatch.type === 'exec' && typeof dispatch.output === 'string') {
    render(dispatch.output)

    return true
  }

  if (dispatch.type === 'send' && typeof dispatch.message === 'string' && dispatch.message.trim()) {
    return await send(dispatch.message)
  }

  return false
}

export async function sendSteeringFallback(options: {
  text: string
  sessionId: string
  storedSessionId: string | null
  foregroundBusy: boolean
  render: (output: string) => void
  submit: (text: string, options: SubmitTextOptions) => Promise<boolean> | boolean
}): Promise<boolean> {
  const queued = queueKickoffIfSessionBusy(options)

  if (queued !== 'idle') {
    options.render(
      queued === 'queued'
        ? 'session busy - message queued to send when the current turn finishes'
        : 'session busy - unable to queue this message'
    )

    return queued === 'queued'
  }

  return await options.submit(options.text, {
    sessionId: options.sessionId,
    storedSessionId: options.storedSessionId,
    attachments: []
  })
}

export async function submitBusyPrompt(options: {
  mode: BusyInputMode | null
  composerScope?: string | null
  attachments?: ComposerAttachment[]
  text: string
  sessionId: string
  storedSessionId: string | null
  foregroundBusy: boolean
  request: SteeringOptions['request']
  update: (
    sessionId: string,
    updater: (state: ClientSessionState) => ClientSessionState,
    storedSessionId: string | null
  ) => unknown
  submit: (text: string, options: SubmitTextOptions) => Promise<boolean> | boolean
  redirect: (text: string) => Promise<boolean> | boolean
}): Promise<boolean> {
  const stored = options.storedSessionId ?? options.sessionId
  const key = resolveComposerSessionKey(stored, $sessions.get()) || stored

  if (options.composerScope !== undefined && options.composerScope !== key) {
    return false
  }

  if (!options.mode) {
    notify({ kind: 'error', message: 'Busy-input settings have not loaded. Your message has been kept.' })

    return false
  }

  // Middleware may add files after the composer chose a text-only busy send.
  const attachments = options.attachments ?? []

  if (attachments.length) {
    if (isTargetSessionBusy($sessionStates.get(), options.sessionId, options.foregroundBusy)) {
      return Boolean(enqueueQueuedPrompt(key, { text: options.text, attachments }))
    }

    return await options.submit(options.text, {
      attachments,
      sessionId: options.sessionId,
      storedSessionId: options.storedSessionId
    })
  }

  const render = (output: string) =>
    options.update(
      options.sessionId,
      state => ({
        ...state,
        messages: [
          ...state.messages,
          {
            id: `busy-notice-${crypto.randomUUID()}`,
            role: 'system',
            parts: [textPart(slashStatusText('/steer', output))]
          }
        ]
      }),
      options.storedSessionId
    )

  const send = (text: string) => sendSteeringFallback({ ...options, text, render })

  try {
    return await routeBusyInput(options.mode, options.text, {
      steer: text => dispatchSteering({ ...options, text, render, send }),
      interrupt: async text => {
        if (await options.redirect(text)) {
          return true
        }

        // Preserve steerDraft's existing redirect-rejection fallback. This is
        // deliberately not used for non-interrupting steering failures.
        const stored = options.storedSessionId ?? options.sessionId
        const key = resolveComposerSessionKey(stored, $sessions.get()) || stored

        return Boolean(enqueueQueuedPrompt(key, { text, attachments: [] }))
      },
      queue: text => sendSteeringFallback({ ...options, text, render: () => undefined })
    })
  } catch {
    notify({
      kind: 'error',
      message: 'Delivery could not be confirmed. Your message has been kept; it was not automatically retried.'
    })

    return false
  }
}

export async function routeBusyInput(
  mode: 'steer' | 'interrupt' | 'queue' | null,
  text: string,
  actions: Record<'steer' | 'interrupt' | 'queue', (text: string) => Promise<boolean> | boolean>
): Promise<boolean> {
  return mode ? await actions[mode](text) : false
}
