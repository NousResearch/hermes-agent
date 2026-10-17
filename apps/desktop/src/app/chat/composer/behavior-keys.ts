import type { KeyboardEvent } from 'react'

import { isSlashCommandText } from '@/lib/chat-runtime'
import { behaviorKeyAction } from '@/store/composer-behavior'

interface SuggestionOptions {
  text: null | string
  blocked: boolean
  readDraft: () => string
  hasAttachments: boolean
  decide: (status: 'consumed' | 'dismissed') => null | string
  consume: (text: string) => void
  markConsumed: () => void
}

export function handleBehaviorSuggestionKey(event: KeyboardEvent<HTMLDivElement>, options: SuggestionOptions): boolean {
  if (options.blocked || !options.text || options.hasAttachments) {
    return false
  }

  if (event.ctrlKey || event.metaKey || event.altKey || event.shiftKey) {
    return false
  }

  if (event.key !== 'Tab' && event.key !== 'Escape') {
    return false
  }

  if (options.readDraft().trim().length > 0) {
    return false
  }

  const accepted = options.decide(event.key === 'Tab' ? 'consumed' : 'dismissed')

  if (!accepted) {
    return false
  }

  event.preventDefault()
  options.markConsumed()

  if (event.key === 'Tab') {
    options.consume(accepted)
  }

  return true
}

interface EnterOptions {
  busy: boolean
  disabled: boolean
  inputDisabled: boolean
  nativeSubmit: boolean
  readDraft: () => string
  syncDraft: (text: string) => void
  submit: () => void
  steer: () => void
  queue: () => void
}

export function handleBehaviorEnterKey(event: KeyboardEvent<HTMLDivElement>, options: EnterOptions): boolean {
  if (event.key !== 'Enter' || event.shiftKey) {
    return false
  }

  const action = behaviorKeyAction({
    key: event.key,
    busy: options.busy,
    ctrlKey: event.ctrlKey,
    metaKey: event.metaKey,
    altKey: event.altKey,
    shiftKey: event.shiftKey
  })

  if (action === 'steer') {
    event.preventDefault()

    if (options.disabled || options.inputDisabled || !options.busy) {
      return true
    }

    const text = options.readDraft()
    options.syncDraft(text)

    if (options.nativeSubmit || isSlashCommandText(text)) {
      options.submit()
    } else {
      options.steer()
    }

    return true
  }

  const customQueue =
    action === 'queue' && !event.altKey && !options.nativeSubmit && !isSlashCommandText(options.readDraft())

  if (!event.metaKey && !event.ctrlKey && !customQueue) {
    return false
  }

  event.preventDefault()

  if (options.busy && !options.disabled) {
    options.syncDraft(options.readDraft())
    options.queue()
  }

  return true
}

interface NativeEnterOptions {
  busy: boolean
  disabled: boolean
  readDraft: () => string
  hasAttachments: boolean
  queued: readonly { id: string }[]
  editingId?: string
  drain: () => unknown
  deliver: (id: string) => unknown
  submit: () => void
}

/** Native fallback after completion and extension keys have had first refusal. */
export function handleNativeEnterKey(event: KeyboardEvent<HTMLDivElement>, options: NativeEnterOptions): boolean {
  if (event.key !== 'Enter' || event.shiftKey) {
    return false
  }

  event.preventDefault()
  // React state can lag fast typing/IME; always inspect the current editor.
  const hasPayload = options.readDraft().trim().length > 0 || options.hasAttachments

  if (options.disabled || (!options.busy && (event.metaKey || event.ctrlKey))) {
    return true
  }

  if (!options.busy && !hasPayload && options.queued.length > 0) {
    void options.drain()

    return true
  }

  if (options.busy && !hasPayload) {
    const head = options.queued.find(entry => entry.id !== options.editingId)

    if (head) {
      void options.deliver(head.id)
    }

    return true
  }

  options.submit()

  return true
}
