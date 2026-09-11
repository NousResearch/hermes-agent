import type { ReadableAtom } from 'nanostores'
import { createContext, useContext } from 'react'

import type { ChatMessage } from '@/lib/chat-messages'
import { type ComposerAttachmentScope, mainComposerScope } from '@/store/composer'
import { $activeSessionAwaitingInput } from '@/store/prompts'
import { $messages } from '@/store/session'

import type { ComposerTarget } from './focus'

/**
 * COMPOSER SCOPE — which live composer a ChatBar instance IS. The main chat's
 * ChatBar runs in the default scope (module-level attachment atom, focus-bus
 * target 'main', the active session's awaiting-input edge). A session tile
 * mounts its ChatBar under its own scope, so N composers coexist: separate
 * attachment chips, separate focus/insert routing, separate Esc semantics.
 *
 * Draft TEXT needs no scoping — it lives in each ChatBar's contentEditable +
 * draftRef and stashes per session key (`stashSessionDraft`), which already
 * differs per surface.
 */
export interface ComposerScope {
  /** This scope's "turn parked on user input" edge — gates Esc-to-stop. */
  $awaitingInput: ReadableAtom<boolean>
  attachments: ComposerAttachmentScope
  /** This scope's transcript. Read it imperatively (input-history browse) to
   *  keep streaming out of the composer's renders; subscribe only off-render
   *  (auto-speak) where the reply edge is the whole point. */
  $messages: ReadableAtom<ChatMessage[]>
  /** Focus-bus routing key (`'main'` | `'tile:<id>'`). */
  target: ComposerTarget
}

export const MAIN_COMPOSER_SCOPE: ComposerScope = {
  $awaitingInput: $activeSessionAwaitingInput,
  $messages,
  attachments: mainComposerScope,
  target: 'main'
}

const ComposerScopeContext = createContext<ComposerScope>(MAIN_COMPOSER_SCOPE)

export const ComposerScopeProvider = ComposerScopeContext.Provider

export const useComposerScope = (): ComposerScope => useContext(ComposerScopeContext)

/**
 * Unique identity for one mounted ChatView/composer pair. Session ids cannot
 * fill this role: a fresh chat has no id yet, and the same stored session can
 * be rendered in more than one layout pane. External submit requests pin this
 * surface id at click time so exactly one composer can claim the task.
 */
const ComposerSurfaceContext = createContext<string | null>(null)

export const ComposerSurfaceProvider = ComposerSurfaceContext.Provider

export const useComposerSurfaceId = (): string | null => useContext(ComposerSurfaceContext)
