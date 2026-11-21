import type { ReadableAtom } from 'nanostores'

import { type NativeComposerSubmitResult, requestNativeComposerSubmit } from '@/app/chat/composer/focus'

export interface PluginFocusedSessionOwner {
  connectionId: string
  profile: string
}

export interface PluginSubmitPromptOptions extends PluginFocusedSessionOwner {
  /** Exact runtime id of the focused, already hydrated conversation. */
  sessionId: string
  text: string
}

export type PluginSubmitPromptResult = NativeComposerSubmitResult

/** Confirmed plain text goes through one visible native surface. This door
 * never navigates, edits the draft, dispatches slash commands or retries. */
export const createConfirmedSubmitPrompt =
  (focusedRuntimeId: ReadableAtom<null | string>, focusedOwner: ReadableAtom<PluginFocusedSessionOwner | null>) =>
  async (options: PluginSubmitPromptOptions): Promise<PluginSubmitPromptResult> => {
    const owner = focusedOwner.get()

    if (
      !options ||
      typeof options.text !== 'string' ||
      !options.text.trim() ||
      options.text.trimStart().startsWith('/') ||
      options.sessionId !== focusedRuntimeId.get() ||
      !owner ||
      options.connectionId !== owner.connectionId ||
      options.profile !== owner.profile
    ) {
      return { status: 'rejected' }
    }

    return requestNativeComposerSubmit(options.text, { sessionId: options.sessionId })
  }
