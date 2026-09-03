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
  slashCompletions: 'composer.slashCompletions'
} as const

export interface ComposerDraft {
  text: string
  attachments?: ComposerAttachment[]
  /** What the transcript shows when model-facing context is appended to text. */
  displayText?: string
}

/** Payload of a `composer.middleware` data contribution. */
export interface ComposerMiddleware {
  /** Rewrite (return a draft), pass through (same draft), or cancel (null). */
  handler: (draft: ComposerDraft) => ComposerDraft | null | Promise<ComposerDraft | null>
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
}

/** Payload of a `composer.atCompletions` data contribution — an extra source
 *  merged into the composer's `@` popover ABOVE the path/reference results.
 *  `query` is the text typed after `@` (no leading `@`). Sources must be
 *  fast and synchronous-ish (called per keystroke after the debounce); slow
 *  lookups belong behind the source's own cache. */
export interface ComposerAtCompletionSource {
  provide: (query: string) => ComposerAtCompletionItem[]
}

/** One plugin-owned row in the `/` completion popover. */
export interface ComposerSlashCompletionItem {
  /** Full command inserted into the draft (for example `/unfollowers`). */
  insert: string
  /** Row label; defaults to `insert`. */
  display?: string
  /** Short explanation shown under the command. */
  meta?: string
  /** Optional section heading; defaults to `Plugin commands`. */
  group?: string
}

/** Payload of a `composer.slashCompletions` contribution. Sources are local,
 * synchronous, and filter their own rows against `query` (without `/`). */
export interface ComposerSlashCompletionSource {
  provide: (query: string) => ComposerSlashCompletionItem[]
}

/** Plugin-provided slash completion sources, reactive to registry edits. */
export function useComposerSlashCompletionSources(): ComposerSlashCompletionSource[] {
  const contributions = useContributions(COMPOSER_AREAS.slashCompletions)

  return useMemo(
    () =>
      contributions
        .map(c => c.data as ComposerSlashCompletionSource)
        .filter(source => typeof source?.provide === 'function'),
    [contributions]
  )
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
 * broken plugin can't eat messages.
 */
export async function runComposerMiddleware(draft: ComposerDraft): Promise<ComposerDraft | null> {
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

      current = next
    } catch {
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
