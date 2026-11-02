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
  /** Owner connection of this scope's session — a profile belongs to ONE
   *  gateway, so voice playback mints against it (cross-connection Bots).
   *  undefined → the active connection. */
  connectionId?: null | string
  /** Owner profile of this scope's session (a Bot tile runs on the Bot's own
   *  profile). Voice playback synthesizes with it; undefined → active profile. */
  profile?: null | string
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

/** The session a composer belongs to, as its `composer.top` contributions see it. */
export interface ComposerSessionInfo {
  /** The session id the composer was mounted for (`null` on a fresh draft). */
  sessionId: null | string
  /** The durable id it is filed under — the runtime id mapped through the session store. */
  storedSessionId: null | string
}

const ComposerSessionContext = createContext<ComposerSessionInfo>({ sessionId: null, storedSessionId: null })

export const ComposerSessionProvider = ComposerSessionContext.Provider

/** Which session this composer is for, read from inside a `composer.top` contribution. A tile's
 *  composer and the main one answer for their OWN session, so a plugin banner never shows one
 *  chat's state above another chat's input. */
export const useComposerSession = (): ComposerSessionInfo => useContext(ComposerSessionContext)
