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

import {
  captureHostComposerAttachmentAuthority,
  type ComposerAttachmentAttempt,
  createComposerAttachmentAttempt
} from './attachment-relay'

export const COMPOSER_AREAS = {
  top: 'composer.top',
  bottom: 'composer.bottom',
  underside: 'composer.underside',
  leading: 'composer.leading',
  actions: 'composer.actions',
  middleware: 'composer.middleware',
  attachments: 'composer.attachments',
  microActions: 'composer.microActions',
  atCompletions: 'composer.atCompletions'
} as const

export interface ComposerDraft {
  text: string
  attachments?: ComposerAttachment[]
}

export interface ComposerPassDisposition {
  disposition: 'pass'
  draft?: ComposerDraft
}

export interface ComposerRejectDisposition {
  disposition: 'reject'
  reason?: string
}

export interface ComposerConsumeDisposition {
  disposition: 'consume'
  receipt?: unknown
}

export type ComposerDisposition = ComposerConsumeDisposition | ComposerPassDisposition | ComposerRejectDisposition
export type ComposerMiddlewareResult = ComposerDisposition | ComposerDraft | null
export type ComposerRunResult =
  | ComposerDraft
  | null
  | (ComposerRejectDisposition & { draft: ComposerDraft })
  | ComposerConsumeDisposition

function closedDataDescriptors(value: unknown): null | Record<PropertyKey, PropertyDescriptor> {
  if (!value || typeof value !== 'object') {
    return null
  }
  let prototype: object | null
  let descriptors: Record<PropertyKey, PropertyDescriptor>
  try {
    prototype = Object.getPrototypeOf(value)
    descriptors = Object.getOwnPropertyDescriptors(value) as Record<PropertyKey, PropertyDescriptor>
  } catch {
    return null
  }
  if (prototype !== null && prototype !== Object.prototype) {
    return null
  }
  if (Reflect.ownKeys(descriptors).some(key => !('value' in descriptors[key]!))) {
    return null
  }

  return descriptors
}

function closedComposerDisposition(value: unknown): ComposerDisposition | null {
  const descriptors = closedDataDescriptors(value)
  const kind = descriptors?.disposition?.value
  if (!descriptors || (kind !== 'pass' && kind !== 'reject' && kind !== 'consume')) {
    return null
  }

  const allowed = kind === 'pass' ? ['disposition', 'draft'] : kind === 'reject' ? ['disposition', 'reason'] : ['disposition', 'receipt']
  const required = ['disposition']
  const keys = Reflect.ownKeys(descriptors)
  if (
    required.some(key => !Object.prototype.hasOwnProperty.call(descriptors, key)) ||
    keys.some(key => typeof key !== 'string' || !allowed.includes(key))
  ) {
    return null
  }
  if (kind === 'pass' && descriptors.draft && (!descriptors.draft.value || typeof descriptors.draft.value !== 'object')) {
    return null
  }
  if (kind === 'reject' && descriptors.reason && typeof descriptors.reason.value !== 'string') {
    return null
  }

  return value as ComposerDisposition
}

export function isComposerTerminalRunResult(
  value: ComposerRunResult
): value is ComposerConsumeDisposition | (ComposerRejectDisposition & { draft: ComposerDraft }) {
  const descriptors = closedDataDescriptors(value)
  const kind = descriptors?.disposition?.value
  if (!descriptors || (kind !== 'consume' && kind !== 'reject')) {
    return false
  }
  const allowed = kind === 'consume' ? ['disposition', 'receipt'] : ['disposition', 'draft', 'reason']
  const keys = Reflect.ownKeys(descriptors)
  return (
    keys.every(key => typeof key === 'string' && allowed.includes(key)) &&
    Object.prototype.hasOwnProperty.call(descriptors, 'disposition') &&
    (kind !== 'reject' || Object.prototype.hasOwnProperty.call(descriptors, 'draft'))
  )
}

/** Payload of a `composer.middleware` data contribution. */
export interface ComposerMiddleware {
  /** Legacy draft/null results remain exact; dispositions add pass/reject/consume. */
  handler: (draft: ComposerDraft) => ComposerMiddlewareResult | Promise<ComposerMiddlewareResult>
}

interface AuthoritativeComposerAttachment {
  data: ComposerAttachment
  origin: ComposerAttachment | null
}

interface AuthoritativeComposerDraft {
  attachments?: AuthoritativeComposerAttachment[]
  text: string
}

const COMPOSER_ATTACHMENT_KEYS = [
  'id',
  'occurrenceId',
  'kind',
  'label',
  'detail',
  'refText',
  'previewUrl',
  'thumbnailUrl',
  'path',
  'attachedSessionId',
  'uploadState'
] as const
const COMPOSER_ATTACHMENT_KINDS = new Set(['file', 'folder', 'image', 'review', 'terminal', 'url'])
const COMPOSER_DRAFT_KEYS = new Set(['text', 'attachments'])

function copyAttachment(attachment: ComposerAttachment): ComposerAttachment {
  return { ...attachment }
}

function materializeDraft(draft: AuthoritativeComposerDraft): ComposerDraft {
  return {
    text: draft.text,
    ...(draft.attachments ? { attachments: draft.attachments.map(item => copyAttachment(item.data)) } : {})
  }
}

function denseAttachmentArray(value: unknown): unknown[] | null {
  if (!Array.isArray(value) || Object.getPrototypeOf(value) !== Array.prototype) {
    return null
  }
  const descriptors = Object.getOwnPropertyDescriptors(value) as unknown as Record<PropertyKey, PropertyDescriptor>
  const length = descriptors.length?.value
  if (!Number.isSafeInteger(length) || length < 0) {
    return null
  }
  const keys = Reflect.ownKeys(descriptors)
  if (
    keys.length !== length + 1 ||
    keys.some(key => key !== 'length' && (typeof key !== 'string' || !/^\d+$/.test(key) || Number(key) >= length))
  ) {
    return null
  }
  const items: unknown[] = []
  for (let index = 0; index < length; index += 1) {
    const descriptor = descriptors[String(index)]
    if (!descriptor || !('value' in descriptor)) {
      return null
    }
    items.push(descriptor.value)
  }

  return items
}

function safeInheritedDispositionPrototype(value: object): boolean {
  let prototype: object | null
  try {
    prototype = Object.getPrototypeOf(value)
  } catch {
    return false
  }
  if (!prototype) {
    return false
  }
  const parent = Object.getPrototypeOf(prototype)
  if (parent !== Object.prototype && parent !== null) {
    return false
  }
  const descriptors = Object.getOwnPropertyDescriptors(prototype) as Record<PropertyKey, PropertyDescriptor>
  const keys = Reflect.ownKeys(descriptors)
  return keys.length === 1 && keys[0] === 'disposition' && 'value' in descriptors.disposition!
}

function validatedAttachment(
  value: unknown,
  attempt: ComposerAttachmentAttempt
): AuthoritativeComposerAttachment | null {
  const descriptors = closedDataDescriptors(value)
  if (!descriptors) {
    return null
  }
  const keys = Reflect.ownKeys(descriptors)
  if (
    keys.some(
      key =>
        typeof key !== 'string' ||
        !COMPOSER_ATTACHMENT_KEYS.includes(key as (typeof COMPOSER_ATTACHMENT_KEYS)[number])
    ) ||
    !Object.prototype.hasOwnProperty.call(descriptors, 'id') ||
    !Object.prototype.hasOwnProperty.call(descriptors, 'kind') ||
    !Object.prototype.hasOwnProperty.call(descriptors, 'label')
  ) {
    return null
  }
  if (
    typeof descriptors.id!.value !== 'string' ||
    !descriptors.id!.value ||
    typeof descriptors.label!.value !== 'string' ||
    !descriptors.label!.value ||
    typeof descriptors.kind!.value !== 'string' ||
    !COMPOSER_ATTACHMENT_KINDS.has(descriptors.kind!.value)
  ) {
    return null
  }
  for (const key of COMPOSER_ATTACHMENT_KEYS) {
    if (!descriptors[key] || key === 'id' || key === 'kind' || key === 'label') {
      continue
    }
    const field = descriptors[key]!.value
    if (typeof field !== 'string' || (key === 'uploadState' && field !== 'uploading' && field !== 'error')) {
      return null
    }
  }
  const origin = attempt.originForUnchangedView(value as ComposerAttachment)
  if (descriptors.path && !origin) {
    return null
  }
  const data: Record<string, unknown> = {}
  for (const key of COMPOSER_ATTACHMENT_KEYS) {
    if (descriptors[key]) {
      data[key] = descriptors[key]!.value
    }
  }

  return { data: data as unknown as ComposerAttachment, origin }
}

function adoptDraft(
  value: unknown,
  attempt: ComposerAttachmentAttempt,
  options: { allowInheritedDisposition?: boolean } = {}
): AuthoritativeComposerDraft | null {
  if (!value || typeof value !== 'object') {
    return null
  }
  let prototype: object | null
  let descriptors: Record<PropertyKey, PropertyDescriptor>
  try {
    prototype = Object.getPrototypeOf(value)
    descriptors = Object.getOwnPropertyDescriptors(value) as Record<PropertyKey, PropertyDescriptor>
  } catch {
    return null
  }
  if (
    prototype !== null &&
    prototype !== Object.prototype &&
    !(options.allowInheritedDisposition && safeInheritedDispositionPrototype(value))
  ) {
    return null
  }
  const keys = Reflect.ownKeys(descriptors)
  if (
    keys.some(key => typeof key !== 'string' || !COMPOSER_DRAFT_KEYS.has(key)) ||
    keys.some(key => !('value' in descriptors[key]!)) ||
    !Object.prototype.hasOwnProperty.call(descriptors, 'text') ||
    typeof descriptors.text!.value !== 'string'
  ) {
    return null
  }
  let attachments: AuthoritativeComposerAttachment[] | undefined
  if (descriptors.attachments) {
    const items = denseAttachmentArray(descriptors.attachments.value)
    if (!items) {
      return null
    }
    attachments = []
    for (const item of items) {
      const attachment = validatedAttachment(item, attempt)
      if (!attachment) {
        return null
      }
      attachments.push(attachment)
    }
  }

  return {
    text: descriptors.text.value,
    ...(attachments ? { attachments } : {})
  }
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
export async function runComposerMiddleware(draft: ComposerDraft): Promise<ComposerRunResult> {
  const hostAttachments = draft.attachments ?? []
  const hostAttachmentAuthority = captureHostComposerAttachmentAuthority(hostAttachments)
  let current: AuthoritativeComposerDraft = {
    text: draft.text,
    ...(draft.attachments
      ? { attachments: hostAttachments.map(attachment => ({ data: attachment, origin: attachment })) }
      : {})
  }
  let invoked = false

  for (const contribution of registry.getArea(COMPOSER_AREAS.middleware)) {
    const middleware = contribution.data as ComposerMiddleware | undefined

    if (!middleware?.handler) {
      continue
    }

    invoked = true
    const authoritativeAttachments = current.attachments ?? []
    const attachmentAttempt = createComposerAttachmentAttempt(
      hostAttachmentAuthority,
      authoritativeAttachments.map(item => item.data),
      authoritativeAttachments.map(item => item.origin)
    )
    const attemptDraft: ComposerDraft = {
      text: current.text,
      ...(current.attachments ? { attachments: attachmentAttempt.attachments } : {})
    }

    try {
      const next = await middleware.handler(attemptDraft)

      if (next === null) {
        return null
      }

      if (typeof next !== 'object') {
        continue
      }

      const disposition = closedComposerDisposition(next)
      if (disposition) {

        if (disposition.disposition === 'pass') {
          const nextDraft = Object.getOwnPropertyDescriptor(disposition, 'draft')?.value as ComposerDraft | undefined
          if (nextDraft) {
            const adopted = adoptDraft(nextDraft, attachmentAttempt)
            if (adopted) {
              current = adopted
            }
          }
          continue
        }

        if (disposition.disposition === 'reject') {
          return {
            disposition: 'reject',
            draft: materializeDraft(current),
            ...(typeof Object.getOwnPropertyDescriptor(disposition, 'reason')?.value === 'string'
              ? { reason: Object.getOwnPropertyDescriptor(disposition, 'reason')!.value as string }
              : {})
          }
        }

        if (disposition.disposition === 'consume') {
          return {
            disposition: 'consume',
            ...(Object.prototype.hasOwnProperty.call(disposition, 'receipt')
              ? { receipt: Object.getOwnPropertyDescriptor(disposition, 'receipt')!.value }
              : {})
          }
        }

        continue
      }

      const adopted = adoptDraft(next, attachmentAttempt, { allowInheritedDisposition: true })
      if (adopted) {
        current = adopted
      }
    } catch {
      // Attempt-local objects are discarded; authoritative state was never exposed.
    } finally {
      attachmentAttempt.release()
    }
  }

  return invoked ? materializeDraft(current) : draft
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
