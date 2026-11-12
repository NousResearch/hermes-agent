/**
 * Composer contribution surface — every seam of the composer is hook-into-able
 * through the SAME registry schema as every other surface (statusbar, titlebar,
 * panes, layouts):
 *
 *   render areas (`render`):  composer.top       — banner strip above the input
 *                             composer.bottom    — row below the input grid
 *                             composer.underside — floating strip BELOW the
 *                                                  whole composer (no chrome)
 *                             composer.leading   — inline after the "+" menu
 *                             composer.actions   — inline before the model pill
 *
 *   data kinds (`data`):      composer.middleware    (ComposerMiddleware)
 *                             composer.attachments   (ComposerAttachmentProvider)
 *                             composer.microActions  (ComposerMicroActionProvider)
 *                             composer.contextMenu   (ComposerContextMenuProvider)
 *
 * Core keeps ownership of the transcript, input, and submit engine — these
 * seams AUGMENT the composer, they never replace it. Middleware runs as an
 * ordered async chain around the app's onSubmit: each handler may rewrite the
 * draft, pass it through, or cancel the send by returning null.
 */

import { useMemo } from 'react'

import { useContributions } from '@/contrib/react/use-contributions'
import { registry } from '@/contrib/registry'
import type { TodoItem } from '@/lib/todos'
import type { ComposerAttachment } from '@/store/composer'
import type { ComposerAction } from '@/store/composer-actions'

import type { ComposerTarget } from './focus'

export const COMPOSER_AREAS = {
  top: 'composer.top',
  bottom: 'composer.bottom',
  underside: 'composer.underside',
  leading: 'composer.leading',
  actions: 'composer.actions',
  middleware: 'composer.middleware',
  attachments: 'composer.attachments',
  microActions: 'composer.microActions',
  atCompletions: 'composer.atCompletions',
  modelPill: 'composer.modelPill',
  contextMenu: 'composer.contextMenu'
} as const

/** What a `composer.contextMenu` provider sees when the user right-clicks the
 *  composer: the word under the caret and the selection INSIDE the composer,
 *  nothing else — no element, no range, no other field's text. Every field
 *  is a standing compatibility promise to the plugins reading it. */
export interface ComposerContextMenuContext {
  /** Selected composer text at the click ('' when the caret is collapsed). */
  readonly selection: string
  /** The word under the caret (or at the selection's start), or null. */
  readonly word: null | string
}

/** Handed to an item's `onSelect` after the menu has closed and the composer
 *  has the caret back. */
export interface ComposerContextMenuAction extends ComposerContextMenuContext {
  /** Replace the right-clicked `word` with `text` (one line). Works once, and
   *  only while the composer still holds that word where it was — false
   *  otherwise, never an edit somewhere else. */
  replaceWord: (text: string) => boolean
}

/** One row the app renders for a plugin in the composer's context menu. */
export interface ComposerContextMenuItem {
  /** Plain text; control characters are stripped and it is clamped. */
  label: string
  disabled?: boolean
  onSelect: (action: ComposerContextMenuAction) => void
}

/** Payload of a `composer.contextMenu` data contribution. `items` runs once
 *  per right-click on the composer; return `[]` for "nothing from me". The
 *  app renders each plugin's rows as its own section (plugins ordered by id,
 *  at most six rows each) between its own spell-check rows and the edit verbs. */
export interface ComposerContextMenuProvider {
  items: (context: ComposerContextMenuContext) => ComposerContextMenuItem[]
}

/** Where a send came from, as the app saw it. Set by the app on every
 *  middleware call and re-stamped after each handler, so a middleware can
 *  read it but never forge it for the next one. Every field is a standing
 *  compatibility promise to the plugins reading it. */
export interface ComposerSendOrigin {
  /** `typed` = the user's own draft sent from this composer (Enter / Send);
   *  `queued` = a queue drain or busy-time queue admission; `voice` = a
   *  spoken turn; `hidden` = machine text that renders no bubble; `other` =
   *  everything else (external submit requests, steer fallbacks, …). */
  readonly kind: 'hidden' | 'other' | 'queued' | 'typed' | 'voice'
  /** The composer that sent it: `main` or a tile's target. */
  readonly target: ComposerTarget
  /** The composer scope (stored session id) the draft was loaded under;
   *  null = a fresh chat with no session yet; undefined = not a composer send. */
  readonly composerScope?: null | string
}

export interface ComposerDraft {
  text: string
  attachments?: ComposerAttachment[]
  /** Absent on direct `runComposerMiddleware` calls without send context. */
  readonly origin?: ComposerSendOrigin
}

/** Payload of a `composer.middleware` data contribution. */
export interface ComposerMiddleware {
  /** Rewrite (return a draft), pass through (same draft), or cancel (null). */
  handler: (draft: ComposerDraft) => ComposerDraft | null | Promise<ComposerDraft | null>
  /** A throw from this handler cancels the send instead of passing it
   *  through. For middleware whose job is to gate what is sent (review,
   *  approval): an unexpected failure there must never let the draft out. */
  failClosed?: boolean
}

/** One row a `composer.atCompletions` source offers for the current query. */
export interface ComposerAtCompletionItem {
  /** Text inserted into the draft when picked (e.g. `@researcher`). */
  insert: string
  /** Row label; defaults to `insert`. */
  display?: string
  /** Secondary line (e.g. "Bot · Homelab"). */
  meta?: string
  /** Icon slug understood by the completion popover; defaults to 'simple'. */
  icon?: string
  /** Other `@handle` texts that resolve to the SAME target as `insert` —
   *  e.g. a bot's raw profile name alongside the title slug it is listed
   *  under. The `@` popover drops a gateway row whose handle is claimed
   *  here, so one routable identity is never offered twice under two
   *  labels. Only claim handles this row really owns. */
  handles?: string[]
}

/** Payload of a `composer.atCompletions` data contribution — an extra source
 *  merged into the composer's `@` popover ABOVE the path/reference results.
 *  `query` is the text typed after `@` (no leading `@`). Sources must be
 *  fast and synchronous-ish (called per keystroke after the debounce); slow
 *  lookups belong behind the source's own cache. */
export interface ComposerAtCompletionSource {
  provide: (query: string) => ComposerAtCompletionItem[]
}

export interface ComposerAttachmentContext {
  insertText: (text: string) => void
}

/** Payload of a `composer.attachments` data contribution — an entry in the
 *  composer's "+" attach menu. */
export interface ComposerAttachmentProvider {
  label: string
  /** Codicon name for the menu row. Defaults to `plug`. */
  icon?: string
  run: (ctx: ComposerAttachmentContext) => void | Promise<void>
}

/**
 * Run the ordered middleware chain over a draft. Contributions execute in
 * registry order (`order`, then registration order); the first `null` wins
 * and cancels the send. A throwing handler is treated as pass-through so a
 * broken plugin can't eat messages — unless it declared `failClosed`, in
 * which case the throw cancels the send (the caller restores the draft).
 */
export async function runComposerMiddleware(draft: ComposerDraft): Promise<ComposerDraft | null> {
  const origin = draft.origin
  let current = draft

  for (const contribution of registry.getArea(COMPOSER_AREAS.middleware)) {
    const middleware = contribution.data as ComposerMiddleware | undefined

    if (!middleware?.handler) {
      continue
    }

    try {
      const next = await middleware.handler(current)

      if (next === null) {
        return null
      }

      current = origin ? { ...next, origin } : next
    } catch {
      if (middleware.failClosed) {
        return null
      }
      // Pass-through: a faulty middleware must never swallow the message.
    }
  }

  return current
}

/** Attach-menu entries contributed by plugins/core, with stable render keys. */
export function useComposerAttachmentProviders(): Array<ComposerAttachmentProvider & { key: string }> {
  return useContributions(COMPOSER_AREAS.attachments)
    .map(c => ({ key: `${c.source ?? 'core'}:${c.id}`, ...(c.data as ComposerAttachmentProvider) }))
    .filter(p => Boolean(p.label && p.run))
}

/**
 * Payload of a `composer.microActions` data contribution — the pill strip at
 * the top of the composer's overlay lane.
 *
 * `resolve` is called with the live session context and returns the badges to
 * show right now, or `[]` for "nothing from me". Returning a list rather than
 * a static badge is what lets a provider be conditional ("only while idle",
 * "only with unfinished tasks") without a reactive `when()`, which the
 * registry deliberately doesn't offer.
 */
export interface ComposerMicroActionProvider {
  resolve: (ctx: ComposerMicroActionContext) => ComposerAction[]
}

/** What a micro-action provider gets to branch on. Deliberately small: every
 *  field here is a standing compatibility promise to the plugins using it. */
export interface ComposerMicroActionContext {
  /** A turn is currently running in this session. */
  busy: boolean
  sessionId: string
  /** Live todo list for the session (empty when there is none). */
  todos: readonly TodoItem[]
}

/** Micro-action providers, memoised against the registry's own stable
 *  snapshot — the strip re-resolves on every composer render, so a fresh array
 *  here would defeat that. */
export function useComposerMicroActionProviders(): ComposerMicroActionProvider[] {
  const contributions = useContributions(COMPOSER_AREAS.microActions)

  return useMemo(
    () => contributions.map(c => c.data as ComposerMicroActionProvider).filter(p => typeof p?.resolve === 'function'),
    [contributions]
  )
}

/** What a model-pill label provider gets to branch on. Deliberately small:
 *  every field here is a standing compatibility promise to the plugins using it. */
export interface ComposerModelPillContext {
  /** The model slug the pill would show. */
  model: string
  /** The session's live reasoning effort ('' when the model has none). */
  reasoningEffort: string
  /** Floating-composer mode renders only the chevron; providers are not consulted. */
  compact: boolean
}

/** Payload of a `composer.modelPill` data contribution — the pill's label text.
 *  Return the label to show, or `null` to let the next provider (then the core
 *  label) win. The pill keeps its chrome and menu; only the label changes. */
export interface ComposerModelPillProvider {
  label: (ctx: ComposerModelPillContext) => null | string
}

/** The first provider-supplied pill label, or `null` for the core label. A
 *  throwing provider is treated as declining — a broken plugin can't blank the
 *  pill. */
export function useComposerModelPillLabel({
  compact,
  model,
  reasoningEffort
}: ComposerModelPillContext): null | string {
  const contributions = useContributions(COMPOSER_AREAS.modelPill)

  // Memoised on the primitive fields (the caller builds a fresh ctx object
  // every render) so providers run only when the registry or the pill's
  // inputs actually change — a plugin's label() must not run per keystroke.
  return useMemo(() => {
    if (compact) {
      return null
    }

    const ctx: ComposerModelPillContext = { compact, model, reasoningEffort }

    for (const contribution of contributions) {
      const provider = contribution.data as ComposerModelPillProvider | undefined

      try {
        const label = provider?.label?.(ctx)

        // Only a non-empty string is a label. ModelPill renders the value
        // straight into JSX with no error boundary, so an object/array/number
        // from a plugin would throw and blank the composer — treat it as declining.
        if (typeof label === 'string' && label.trim() !== '') {
          return label
        }
      } catch {
        // Decline on throw: the next provider, then the core label, wins.
      }
    }

    return null
  }, [contributions, compact, model, reasoningEffort])
}
