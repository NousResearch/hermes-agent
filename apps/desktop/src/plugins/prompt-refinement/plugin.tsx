/**
 * Prompt Refinement — rewrite a draft with a side model and let the user
 * approve, edit, keep or cancel it before anything is sent.
 *
 *   Off              sends pass straight through (the shipped default).
 *   Approval-First   Enter refines the draft and opens the review drawer;
 *                    nothing is sent until the user approves, edits, sends
 *                    the original or cancels.
 *   Automatic        Enter refines and sends; it stops for the user only when
 *                    refinement fails or would drop something.
 *
 * Both run in one fail-closed `composer.middleware`. The composer's Refine
 * pill switches mode, depth and style.
 *
 * The refinement itself is a stateless `llm.oneshot` call (no session, no
 * history, no tools, no skills), routed to the backend that owns the chat:
 * isolated from the agent's execution, tool calls and delegation.
 *
 * Layout: types / settings / prompt / preservation / engine / review-store /
 * middleware / scope-guard are plain modules with their own tests;
 * cache / chat-context are the reuse cache and the opt-in chat excerpt;
 * review-drawer, refine-control and settings-page are the UI; host-bridge is
 * the one file that talks to the app (through `@hermes/plugin-sdk` only). A
 * Python half
 * (`plugins/prompt-refinement/`) only registers the `prompt_refinement`
 * auxiliary model slot.
 */

import { COMPOSER_AREAS, type HermesPlugin, host } from '@hermes/plugin-sdk'

import { cacheKey, createRefinementCache } from './cache'
import { recentContext } from './chat-context'
import { runRefinement } from './engine'
import { focusedChat, focusedRoute, focusedRuntimeSessionId, isSlashCommand, oneShotCaller } from './host-bridge'
import { PLUGIN_ID, PROMPT_REFINEMENT_LOCALES } from './i18n'
import { createRefinementMiddleware, type MiddlewareDeps } from './middleware'
import { createRefineControl } from './refine-control'
import { createReviewDrawer } from './review-drawer'
import { createReviewStore } from './review-store'
import { createScopeGuard } from './scope-guard'
import { createSettingsStore, SETTINGS_KEY } from './settings'
import { createSettingsPage } from './settings-page'
import type { RefineRequest } from './types'

/** Run after every other middleware, so what the user reviews is what is sent. */
const MIDDLEWARE_ORDER = 10_000

const SEND_NOTICES: Record<Parameters<MiddlewareDeps['notify']>[0], { kind: 'info' | 'warning'; key: string }> = {
  'scope-changed': { kind: 'info', key: 'notify.scopeChanged' },
  'no-host': { kind: 'warning', key: 'notify.noHost' },
  busy: { kind: 'info', key: 'notify.busy' },
  error: { kind: 'warning', key: 'notify.error' }
}

const plugin: HermesPlugin = {
  id: PLUGIN_ID,
  name: 'Prompt Refinement',
  description: 'Review an improved version of your prompt before it is sent. Off until you choose a mode.',
  register(ctx) {
    ctx.i18n.register(PROMPT_REFINEMENT_LOCALES)
    const t = ctx.i18n.t
    const settings = createSettingsStore(ctx.storage)
    const reviews = createReviewStore()
    const ReviewDrawer = createReviewDrawer(reviews)
    const SettingsPage = createSettingsPage(settings)

    // Another window changed the preference: follow it.
    ctx.addEventListener(window, 'storage', event => {
      if ((event as StorageEvent).key?.endsWith(SETTINGS_KEY)) {
        settings.reload()
      }
    })

    // Disabling or reloading the plugin mid-review cancels it: the composer
    // restores the draft, nothing is sent.
    ctx.onDispose(() => reviews.cancelActive('unloaded'))

    // The same request on the same route (connection, profile, and the chat's
    // model when borrowed) is answered from the reuse cache for a few minutes;
    // Try again always asks afresh.
    const cache = createRefinementCache()
    ctx.onDispose(() => cache.clear())

    const refine = (request: RefineRequest, signal: AbortSignal, attempt: number) => {
      const modelSource = settings.get().modelSource
      const borrowSessionModel = modelSource === 'chat'
      const routeSessionId = focusedRuntimeSessionId()
      const key = cacheKey(request, { ...focusedRoute(), modelSource, sessionId: routeSessionId })

      return cache.run(key, attempt, () =>
        runRefinement(request, oneShotCaller({ routeSessionId, borrowSessionModel }), { signal })
      )
    }

    const middleware = createRefinementMiddleware({
      settings: settings.get,
      review: reviews.request,
      refine: (request, _draft, signal, attempt) => refine(request, signal, attempt),
      holdComposer: target => host.composer.hold(target),
      contextFor: recentContext,
      guardScope: (_draft, onDrift) => createScopeGuard(focusedChat, onDrift),
      isSlashCommand,
      cancelActiveReview: () => reviews.cancelActive('scope-changed'),
      notify: event => host.notify({ kind: SEND_NOTICES[event].kind, message: t(SEND_NOTICES[event].key) })
    })

    const RefineControl = createRefineControl(settings)

    ctx.registerMany([
      { id: 'review', area: COMPOSER_AREAS.middleware, order: MIDDLEWARE_ORDER, data: middleware },
      // Slides up out of the typing box of the composer the review belongs to.
      { id: 'review-drawer', area: COMPOSER_AREAS.top, render: () => <ReviewDrawer /> },
      // The composer's Refine pill: mode, depth and style in one click.
      { id: 'refine-control', area: COMPOSER_AREAS.actions, render: () => <RefineControl /> }
    ])

    ctx.registerSettingsPage({
      id: 'settings',
      title: t('settings.title'),
      icon: 'sparkle',
      render: () => <SettingsPage />
    })
  }
}

export default plugin
