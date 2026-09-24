/**
 * Composer focus + external-insert bus.
 *
 * Mutations from outside the composer (sidebar attach, drag drop, terminal
 * Cmd+L, preview console, etc.) dispatch through here. Each composer subscribes
 * and routes the work back into its own ref/state.
 *
 * `dispatch` defers to a macrotask so synchronous click/keydown handlers
 * (react-arborist row focus, picker `node.select()`) finish first and don't
 * steal focus from the composer effect.
 */

import { isElementInHiddenPane, queryAllVisible, queryVisible } from '@/components/pane-shell/pane-visibility'
import { $hoveredTreeGroup } from '@/components/pane-shell/tree/store'

import { $floatingComposerOwner } from './floating-state'
import type { InlineRefInput } from './inline-refs'
import { RICH_INPUT_SLOT } from './rich-editor'

/** Composer routing key. The main chat is `'main'`, the edit composer
 *  `'edit'`; scoped composers (session tiles) use `'tile:<id>'`. */
export type ComposerTarget = 'edit' | 'main' | (string & {})
export type ComposerInsertMode = 'block' | 'inline' | 'prefix'

export interface FocusDetail {
  target: ComposerTarget
  /** Append after focus (type-to-focus / soft `/`). */
  typeChar?: string
}

interface InsertDetail {
  mode: ComposerInsertMode
  target: ComposerTarget
  text: string
  /** Present when the caller wants an acknowledgement (plugin SDK
   *  `host.composer.insertText`); the claiming subscriber echoes it back on
   *  {@link INSERT_REPLY_EVENT}. Internal fire-and-forget inserts omit it. */
  token?: number
}

interface InsertRefsDetail {
  refs: InlineRefInput[]
  target: ComposerTarget
}

/** Reply the claiming surface sends for a tokened insert. */
interface InsertReplyDetail {
  ok: boolean
  token: number
}

interface AttachImagesDetail {
  blobs: Blob[]
  target: ComposerTarget
}

const FOCUS_EVENT = 'hermes:composer-focus'
const INSERT_EVENT = 'hermes:composer-insert'
const INSERT_REPLY_EVENT = 'hermes:composer-insert-reply'
const ATTACH_IMAGES_EVENT = 'hermes:composer-attach-images'
const INSERT_REFS_EVENT = 'hermes:composer-insert-refs'
const SUBMIT_EVENT = 'hermes:composer-submit'
const VOICE_TOGGLE_EVENT = 'hermes:composer-voice-toggle'
const DICTATION_EVENT = 'hermes:composer-dictation'
const MODEL_MENU_EVENT = 'hermes:composer-model-menu'

/** Inline edit composer root — mounted only while a user bubble is being edited. */
export const EDIT_COMPOSER_ROOT = '[data-slot="aui_edit-composer-root"]'

/** Attribute-safe selector fragment. jsdom (vitest) does not ship `CSS.escape`. */
const cssEscape = (value: string): string => {
  if (typeof CSS !== 'undefined' && typeof CSS.escape === 'function') {
    return CSS.escape(value)
  }

  // Our targets are `'main'` / `'edit'` / `'tile:<id>'` — alphanumerics plus `:`
  // and `-`. Escape anything outside that set so a weird id cannot break the
  // attribute selector.
  return value.replace(/[^a-zA-Z0-9_:-]/g, ch => `\\${ch}`)
}

interface SubmitDetail {
  /** Unique mounted composer surface captured at click time. */
  surfaceId: string
  target: ComposerTarget
  text: string
  /** `hidden` types the persisted user row so no bubble renders — the
   *  off-screen path for widget intents. Omit for normal visible sends. */
  displayKind?: 'hidden'
}

let activeTarget: ComposerTarget = 'main'

/**
 * The chat surface currently on screen (`data-composer-target` hung off each
 * ChatView). Inactive tabs stay mounted with `data-pane-hidden`, so this uses
 * the same visibility policy as every other document-wide surface lookup.
 */
const visibleChatTarget = (): ComposerTarget | null => {
  if (typeof document === 'undefined') {
    return null
  }

  const surface = queryVisible<HTMLElement>('[data-composer-target]')
  const target = surface?.dataset.composerTarget

  return target ? (target as ComposerTarget) : null
}

/** True when `target` still has a live, on-screen subscriber. */
const targetIsReachable = (target: ComposerTarget): boolean => {
  if (typeof document === 'undefined') {
    return true
  }

  // The edit composer is an in-thread overlay, not a chat surface — it never
  // stamps `data-composer-target`. While its root is mounted it still owns the
  // bus; once it tears down the claim is dead.
  if (target === 'edit') {
    return Boolean(document.querySelector(EDIT_COMPOSER_ROOT))
  }

  // Exact match on a VISIBLE surface. Background keep-alive tabs carry the same
  // `data-composer-target` but sit under `data-pane-hidden`, so queryVisible
  // filters them out.
  if (queryVisible(`[data-composer-target="${cssEscape(target)}"]`)) {
    return true
  }

  // A different chat surface is on screen → this claim is buried or gone.
  // (A claim with zero stamped surfaces yet — first paint, pure-unit tests —
  // keeps the marked key until the DOM contradicts it.)
  if (queryVisible('[data-composer-target]')) {
    return false
  }

  return true
}

/**
 * The composer `'active'` should route to right now.
 *
 * The cached claim (`activeTarget`) wins while its surface is still on screen.
 * Tab stacks keep inactive panes mounted, so focusing a tile then clicking the
 * main tab leaves `activeTarget` pointing at a buried composer — with no
 * subscriber on the visible surface, every type-to-focus keystroke is
 * preventDefault'd and dropped. Heal to the visible chat surface (or main)
 * whenever the claim is off-screen or gone, and keep the cache honest so Esc /
 * voice / soft `/` agree with the keyboard path.
 */
const resolveActive = (): ComposerTarget => {
  const owner = $floatingComposerOwner.get()?.target

  if (owner && (activeTarget !== 'edit' || !targetIsReachable('edit'))) {
    activeTarget = owner

    return owner
  }

  if (targetIsReachable(activeTarget)) {
    return activeTarget
  }

  const visible = visibleChatTarget() ?? 'main'

  activeTarget = visible

  return visible
}

const resolve = (target: ComposerTarget | 'active') => (target === 'active' ? resolveActive() : target)

const dispatch = <T>(name: string, detail: T) => {
  if (typeof window === 'undefined') {
    return
  }

  window.setTimeout(() => window.dispatchEvent(new CustomEvent<T>(name, { detail })), 0)
}

/** Submit is the one bus mutation that must preserve the chat visible at click
 * time. Deferring it lets a parent click handler/tab reveal switch the active
 * keep-alive pane before subscribers run, so the task is dropped or claimed by
 * another composer. Other bus events intentionally defer for focus restoration.
 */
const dispatchNow = <T>(name: string, detail: T) => {
  if (typeof window !== 'undefined') {
    window.dispatchEvent(new CustomEvent<T>(name, { detail }))
  }
}

/** Unique identity for the visible composer surface addressed by a submit. */
export const getVisibleComposerSurfaceId = (target: ComposerTarget): string | null => {
  if (typeof document === 'undefined') {
    return null
  }

  const surface = queryVisible<HTMLElement>(`[data-composer-target="${cssEscape(target)}"]`)

  return surface?.dataset.composerSurfaceId || null
}

const composerSurfaceIsVisible = (target: ComposerTarget, surfaceId: string): boolean => {
  if (typeof document === 'undefined') {
    return false
  }

  return queryAllVisible<HTMLElement>(`[data-composer-target="${cssEscape(target)}"]`).some(
    surface => surface.dataset.composerSurfaceId === surfaceId
  )
}

const subscribe = <T>(name: string, handler: (detail: T) => void) => {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const listener = (event: Event) => {
    const detail = (event as CustomEvent<T>).detail

    if (detail) {
      handler(detail)
    }
  }

  window.addEventListener(name, listener)

  return () => window.removeEventListener(name, listener)
}

export const markActiveComposer = (target: ComposerTarget) => {
  activeTarget = target
}

/** Hand the routing key back when a composer unmounts, so `'active'` can never
 *  resolve to a composer that no longer has a subscriber — such a request is
 *  dispatched and then dropped by every mounted composer's target filter, and
 *  nothing re-marks the active composer on its own.
 *
 *  Guarded on identity: a composer unmounting AFTER another one claimed the key
 *  (closing a background tile, a deferred edit-close cleanup) must not steal it
 *  from the live claimant. Falls through to {@link resolveActive} when the
 *  caller's surface is buried rather than gone, so closing on a tab switch that
 *  already re-fronted another chat surfaces there immediately. */
export const releaseActiveComposer = (target: ComposerTarget) => {
  if (activeTarget !== target) {
    return
  }

  // Prefer the visible chat surface over a hard `'main'` default — releasing a
  // closed tile while another tile is fronted should land there, not the
  // (possibly buried) workspace tab.
  activeTarget = visibleChatTarget() ?? 'main'
}

/** The composer that last held focus — the target `'active'` resolves to.
 *  Used by broadcast listeners (voice, Esc-to-stop) to act on exactly one.
 *  Heals a stale claim the same way {@link requestComposerFocus} does, so Esc
 *  and type-to-focus never disagree after a tab switch left the bus pointing at
 *  a keep-alive-mounted background composer. */
export const getActiveComposer = (): ComposerTarget => resolveActive()

export const requestComposerFocus = (
  target: ComposerTarget | 'active' = 'active',
  { typeChar }: { typeChar?: string } = {}
) => {
  const detail = { target: resolve(target), typeChar }
  const owner = $floatingComposerOwner.get()

  // A first character must land before subsequent native input events, not
  // behind a timer that can reorder fast typing or a pane handoff.
  if (typeChar) {
    dispatchNow<FocusDetail>(FOCUS_EVENT, detail)
  } else if (typeof window !== 'undefined') {
    window.setTimeout(() => {
      if (!owner || $floatingComposerOwner.get() === owner) {
        dispatchNow<FocusDetail>(FOCUS_EVENT, detail)
      }
    }, 0)
  }
}

export const requestComposerInsert = (
  text: string,
  { mode = 'block', target = 'active' }: { mode?: ComposerInsertMode; target?: ComposerTarget | 'active' } = {}
) => {
  const trimmed = text.trim()

  if (!trimmed) {
    return
  }

  dispatch<InsertDetail>(INSERT_EVENT, { mode, target: resolve(target), text: trimmed })
}

/** Acked insert for the plugin SDK (`host.composer.insertText`).
 *
 *  Same trim + deferred dispatch as {@link requestComposerInsert}, plus a
 *  token the claiming surface echoes on {@link INSERT_REPLY_EVENT}. Resolves
 *  false when the text trims to nothing, when no mounted surface claims the
 *  address, or when none answers within the settle window — so a plugin gets
 *  the same fail-closed success signal as `setDraft`/`submit` instead of a
 *  silent no-op. Internal callers keep the fire-and-forget form above. */
export const requestComposerInsertAcked = (
  text: string,
  { mode = 'block', target = 'active' }: { mode?: ComposerInsertMode; target?: ComposerTarget | 'active' } = {}
): Promise<boolean> => {
  const trimmed = text.trim()
  const pending = insertReplies

  if (!trimmed || !pending) {
    return Promise.resolve(false)
  }

  const token = ++insertToken
  const resolvedTarget = resolve(target)

  return new Promise<boolean>(resolve => {
    pending.set(token, resolve)

    dispatch<InsertDetail>(INSERT_EVENT, { mode, target: resolvedTarget, text: trimmed, token })

    // No claimant (unmounted address, input disabled, mid-teardown) must not
    // strand the promise: settle false and drop the slot.
    window.setTimeout(() => {
      if (pending.get(token) === resolve) {
        pending.delete(token)
        resolve(false)
      }
    }, INSERT_REPLY_TIMEOUT_MS)
  })
}

/** Subscriber-side ack for {@link requestComposerInsertAcked}: the surface that
 *  appended the text reports success so the plugin's promise settles. A token-less
 *  insert (the internal bus) is a no-op here. */
export const ackComposerInsert = (token: number | undefined, ok: boolean) => {
  if (token === undefined || typeof window === 'undefined') {
    return
  }

  window.dispatchEvent(new CustomEvent<InsertReplyDetail>(INSERT_REPLY_EVENT, { detail: { ok, token } }))
}

const INSERT_REPLY_TIMEOUT_MS = 50

let insertToken = 0

const insertReplies = typeof window === 'undefined' ? null : new Map<number, (ok: boolean) => void>()

if (typeof window !== 'undefined') {
  window.addEventListener(INSERT_REPLY_EVENT, event => {
    const reply = (event as CustomEvent<InsertReplyDetail>).detail
    const resolve = reply && insertReplies?.get(reply.token)

    if (resolve) {
      insertReplies?.delete(reply.token)
      resolve(reply.ok === true)
    }
  })
}

export const onComposerFocusRequest = (handler: (detail: FocusDetail) => void) =>
  subscribe<FocusDetail>(FOCUS_EVENT, handler)

export const onComposerInsertRequest = (handler: (detail: InsertDetail) => void) =>
  subscribe<InsertDetail>(INSERT_EVENT, handler)

/** Attach image blobs to a composer's attachment set — the unfocused-paste
 *  path (paste-to-focus) hands clipboard images over here. The edit composer
 *  takes no attachments (its own paste path ignores images), so a request
 *  resolving to `'edit'` is dropped by that surface's target filter. */
export const requestComposerAttachImages = (
  blobs: Blob[],
  { target = 'active' }: { target?: ComposerTarget | 'active' } = {}
) => {
  if (blobs.length) {
    dispatch<AttachImagesDetail>(ATTACH_IMAGES_EVENT, { blobs, target: resolve(target) })
  }
}

export const onComposerAttachImagesRequest = (handler: (detail: AttachImagesDetail) => void) =>
  subscribe<AttachImagesDetail>(ATTACH_IMAGES_EVENT, handler)

/** Insert typed ref chips (carrying a display label) into a composer — the
 * structured cousin of {@link requestComposerInsert}, used for session links. */
export const requestComposerInsertRefs = (
  refs: InlineRefInput[],
  { target = 'active' }: { target?: ComposerTarget | 'active' } = {}
) => {
  if (refs.length) {
    dispatch<InsertRefsDetail>(INSERT_REFS_EVENT, { refs, target: resolve(target) })
  }
}

export const onComposerInsertRefsRequest = (handler: (detail: InsertRefsDetail) => void) =>
  subscribe<InsertRefsDetail>(INSERT_REFS_EVENT, handler)

// ── Draft read/write bus (plugin SDK `host.composer`) ─────────────────────
//
// A synchronous request/reply pair over the same CustomEvent bus as the
// mutations above: mounted composers answer for their own sessions, the
// requester times out to `null`/`false` when none does. Deferral is NOT used
// for the request (the reply path is already async for the caller); handlers
// run inline, matching the submit bus's preserve-the-visible-surface rule.

/** Durable or runtime ids a mounted composer answers for (primary: both;
 *  tile: its stored id; plus the queue-edit key the draft stash is keyed by).
 *  `active` addresses the composer the bus currently routes to, whatever
 *  session it holds — the read/write cousin of the mutations' 'active' target. */
export interface DraftRequestDetail {
  token: number
  ids: string[]
  active?: boolean
  /** Set-draft payload; absent on read requests. */
  text?: string
  /** Stamped by the first owning surface so a second owner of the same id
   *  (the primary pane plus a keep-alive tile showing that session) skips it. */
  claimed?: boolean
}

interface DraftReplyDetail {
  ids?: string[]
  ok?: boolean
  text?: null | string
  token: number
}

const GET_DRAFT_EVENT = 'hermes:composer-get-draft'
const SET_DRAFT_EVENT = 'hermes:composer-set-draft'
const DRAFT_REPLY_EVENT = 'hermes:composer-draft-reply'
const DRAFT_REPLY_TIMEOUT_MS = 50

let draftToken = 0

const draftRequests =
  typeof window === 'undefined'
    ? null
    : {
        get: new Map<number, (reply: DraftReplyDetail | null) => void>(),
        set: new Map<number, (reply: DraftReplyDetail | null) => void>()
      }

const requestDraftOnce = (
  channel: 'get' | 'set',
  event: string,
  detail: DraftRequestDetail
): null | Promise<DraftReplyDetail | null> => {
  const pending = draftRequests?.[channel]

  if (!pending) {
    return null
  }

  return new Promise<DraftReplyDetail | null>(resolve => {
    pending.set(detail.token, resolve)

    window.dispatchEvent(new CustomEvent<DraftRequestDetail>(event, { detail }))

    // A subscriber that never replies (unmounted, input disabled, mid-teardown)
    // must not strand the promise: settle null, drop the slot.
    window.setTimeout(() => {
      if (pending.get(detail.token) === resolve) {
        pending.delete(detail.token)
        resolve(null)
      }
    }, DRAFT_REPLY_TIMEOUT_MS)
  })
}

if (typeof window !== 'undefined') {
  window.addEventListener(DRAFT_REPLY_EVENT, event => {
    const reply = (event as CustomEvent<DraftReplyDetail>).detail

    if (!reply?.token) {
      return
    }

    const channel = reply.text === undefined ? 'set' : 'get'
    const pending = draftRequests?.[channel]
    const resolve = pending?.get(reply.token)

    if (resolve) {
      pending?.delete(reply.token)
      resolve(reply)
    }
  })
}

/** Read a mounted composer's LIVE draft (the stash only holds the last
 *  debounced persist). `ids` are the session ids to answer for; the first
 *  relevant surface replies. Resolves null when no mounted composer answers —
 *  callers fall back to `takeSessionDraft` for the persisted copy. */
export const requestComposerGetDraft = (
  ids: string[],
  opts?: { active?: boolean }
): Promise<null | { text: null | string }> => {
  const cleaned = [...new Set(ids.map(id => id?.trim()).filter(Boolean))] as string[]
  const active = opts?.active === true
  const token = ++draftToken

  const promise =
    cleaned.length || active ? requestDraftOnce('get', GET_DRAFT_EVENT, { active, ids: cleaned, token }) : null

  if (!promise) {
    return Promise.resolve(null)
  }

  return promise.then(reply => (reply ? { text: reply.text ?? '' } : null))
}

/** Replace a mounted composer's draft (the app's own paint path — the text
 *  re-renders through `renderComposerContents`, so `@`-ref / `/`-command
 *  tokens hydrate as chips exactly like official paste). Returns false when
 *  no mounted surface answers; never writes another session's composer. */
export const requestComposerSetDraft = (ids: string[], text: string, opts?: { active?: boolean }): Promise<boolean> => {
  const cleaned = [...new Set(ids.map(id => id?.trim()).filter(Boolean))] as string[]
  const active = opts?.active === true
  const token = ++draftToken

  const promise =
    cleaned.length || active ? requestDraftOnce('set', SET_DRAFT_EVENT, { active, ids: cleaned, text, token }) : null

  return promise ? promise.then(reply => reply?.ok === true) : Promise.resolve(false)
}

/** Subscribe one mounted composer to draft read/write requests. `getIds` is
 *  consulted per request (the surface's session identity changes as the user
 *  navigates); `isActive` reports whether the focus bus currently routes to
 *  this composer. An `active` request is answered only by the surface the bus
 *  routes to; an id-addressed request only by a surface owning one of its ids
 *  — the rest stay ignored so N mounted composers coexist on the bus. `read`
 *  answers with the live text; `write` replaces the draft and reports success. */
export const onComposerDraftRequests = (
  address: { getIds: () => string[]; isActive: () => boolean },
  handlers: { read: () => null | string; write: (text: string) => boolean }
) => {
  if (typeof window === 'undefined') {
    return () => undefined
  }

  const listener = (event: Event) => {
    const e = event as CustomEvent<DraftRequestDetail>

    if (!e.detail) {
      return
    }

    // Exactly ONE surface answers a request. `active` belongs to the composer
    // the focus bus routes to — every mounted surface claiming it (the previous
    // behavior) let listener registration order decide instead: with keep-alive
    // tabs in the stack a buried composer answered the read, and a `set`
    // painted onto every mounted draft. An id-addressed request can have two
    // owners (the primary pane and a keep-alive tile showing the same session);
    // the first to see it claims it and the other skips.
    if (e.detail.claimed) {
      return
    }

    if (e.detail.active) {
      if (!address.isActive()) {
        return
      }
    } else {
      const ids = address.getIds()

      if (!e.detail.ids?.some(id => ids.includes(id))) {
        return
      }
    }

    e.detail.claimed = true

    if (e.type === GET_DRAFT_EVENT) {
      window.dispatchEvent(
        new CustomEvent<DraftReplyDetail>(DRAFT_REPLY_EVENT, {
          detail: { text: handlers.read(), token: e.detail.token }
        })
      )
    } else if (e.type === SET_DRAFT_EVENT) {
      const ok = handlers.write(e.detail.text ?? '')

      window.dispatchEvent(
        new CustomEvent<DraftReplyDetail>(DRAFT_REPLY_EVENT, { detail: { ok, token: e.detail.token } })
      )
    }
  }

  window.addEventListener(GET_DRAFT_EVENT, listener)
  window.addEventListener(SET_DRAFT_EVENT, listener)

  return () => {
    window.removeEventListener(GET_DRAFT_EVENT, listener)
    window.removeEventListener(SET_DRAFT_EVENT, listener)
  }
}

/** Submit a prompt through a composer as if the user typed + sent it. Lets
 * external panels (e.g. the review pane's "let the agent ship it" button) hand
 * the agent a task without the user round-tripping through the input. */
export const requestComposerSubmit = (
  text: string,
  {
    displayKind,
    surfaceId: requestedSurfaceId,
    target = 'active'
  }: { displayKind?: 'hidden'; surfaceId?: null | string; target?: ComposerTarget | 'active' } = {}
): boolean => {
  const trimmed = text.trim()

  if (!trimmed) {
    return false
  }

  const resolvedTarget = resolve(target)

  const surfaceId = requestedSurfaceId === undefined ? getVisibleComposerSurfaceId(resolvedTarget) : requestedSurfaceId

  // Fail closed: without an exact visible surface identity, broadcasting a
  // submit could make more than one keep-alive/new-chat composer claim it.
  if (!surfaceId || (requestedSurfaceId !== undefined && !composerSurfaceIsVisible(resolvedTarget, surfaceId))) {
    return false
  }

  dispatchNow<SubmitDetail>(SUBMIT_EVENT, {
    surfaceId,
    target: resolvedTarget,
    text: trimmed,
    ...(displayKind ? { displayKind } : {})
  })

  return true
}

export const onComposerSubmitRequest = (handler: (detail: SubmitDetail) => void) =>
  subscribe<SubmitDetail>(SUBMIT_EVENT, handler)

/** Toggle ONE composer's voice conversation — the `composer.voice` hotkey
 *  (Ctrl+B) reaches the composer that owns voice. Defaults to the active
 *  composer so N tiles don't all flip together. */
export const requestVoiceToggle = (target: ComposerTarget | 'active' = 'active') =>
  dispatch<{ target: ComposerTarget }>(VOICE_TOGGLE_EVENT, { target: resolve(target) })

export const onComposerVoiceToggleRequest = (handler: (target: ComposerTarget) => void) =>
  subscribe<{ target: ComposerTarget }>(VOICE_TOGGLE_EVENT, ({ target }) => handler(target))

/** Start or stop dictation on one composer. Like voice conversation, the
 * rebindable action targets only the active visible composer. */
export const requestComposerDictation = (target: ComposerTarget | 'active' = 'active') =>
  dispatch<{ target: ComposerTarget }>(DICTATION_EVENT, { target: resolve(target) })

export const onComposerDictationRequest = (handler: (target: ComposerTarget) => void) =>
  subscribe<{ target: ComposerTarget }>(DICTATION_EVENT, ({ target }) => handler(target))

/** The chat surface inside the zone the pointer is over, if any. Mirrors the
 *  tab verbs' hover-first targeting (`tabTargetGroupId`, #74447): the model
 *  hotkey lands in the pane you're pointing at without clicking into it first.
 *  Hidden keep-alive tabs are skipped like every document-wide lookup. */
const composerTargetInHoveredZone = (): ComposerTarget | null => {
  const zone = $hoveredTreeGroup.get()

  if (!zone || typeof document === 'undefined') {
    return null
  }

  const surface = queryAllVisible<HTMLElement>('[data-composer-target]').find(
    el => el.closest<HTMLElement>('[data-tree-group]')?.dataset.treeGroup === zone
  )

  return (surface?.dataset.composerTarget as ComposerTarget | undefined) ?? null
}

/** Toggle ONE composer's model menu — the `composer.modelPicker` hotkey.
 *  Targets the pane under the pointer first (the tab-verb convention), then
 *  the active composer. Returns false when no chat surface is on screen at
 *  all (settings, profiles…), so the caller can fall back to the full
 *  model-picker dialog instead of dispatching into the void. */
export const requestModelMenuToggle = (): boolean => {
  if (typeof document !== 'undefined' && !queryVisible('[data-composer-target]')) {
    return false
  }

  dispatch<{ target: ComposerTarget }>(MODEL_MENU_EVENT, {
    target: composerTargetInHoveredZone() ?? resolveActive()
  })

  return true
}

export const onComposerModelMenuRequest = (handler: (target: ComposerTarget) => void) =>
  subscribe<{ target: ComposerTarget }>(MODEL_MENU_EVENT, ({ target }) => handler(target))

/**
 * Focus a composer input across React commit + browser focus restore.
 *
 * The triple-call survives:
 *   - sync: contenteditable already mounted
 *   - rAF:  React just committed a `renderComposerContents` swap
 *   - 0ms:  browser focus reclaim from a click target inside an external panel
 */
export const focusComposerInput = (el: HTMLElement | null) => {
  if (!el) {
    return
  }

  // Skip when already focused: focus() runs the full focusing steps (forcing
  // layout) even on the active element, and during a session switch the DOM is
  // large and dirty — the redundant retries were measurably expensive there.
  // Also skip when another VISIBLE composer holds the caret — a keep-alive
  // remount must not yank typing. A hidden tab that still has DOM focus must
  // not block the pane the user just switched to.
  const owner = $floatingComposerOwner.get()
  const surfaceId = el.closest<HTMLElement>('[data-composer-owner]')?.dataset.composerOwner

  const focus = () => {
    if (owner && (owner !== $floatingComposerOwner.get() || (surfaceId && surfaceId !== owner.id))) {
      return
    }

    if (!el.isConnected || isElementInHiddenPane(el) || document.activeElement === el) {
      return
    }

    const active = document.activeElement

    if (
      active instanceof HTMLElement &&
      active.dataset.slot === RICH_INPUT_SLOT &&
      !isElementInHiddenPane(active) &&
      (!owner || surfaceId !== owner.id)
    ) {
      return
    }

    el.focus({ preventScroll: true })
  }

  focus()
  window.requestAnimationFrame(focus)
  window.setTimeout(focus, 0)
}

/** Drop focus from the main composer input (status-stack chrome, sidebar, etc.).
 *  Skips inactive tabs — they stay mounted, so an unscoped lookup can land on a
 *  background composer and leave the visible one focused. */
export const blurComposerInput = () => {
  const el = queryVisible(`[data-slot="${RICH_INPUT_SLOT}"]`)

  if (el && document.activeElement === el) {
    el.blur()
  }
}
