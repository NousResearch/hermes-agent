import { setModelPreset } from '@/store/model-presets'
import { notifyError } from '@/store/notifications'
import { setCurrentReasoningEffort } from '@/store/session'

/**
 * Shared write path for the composer's reasoning-effort pill and the per-row
 * reasoning submenu in the model picker. Writes the model's preset, optimistically
 * updates the active session's reasoning effort, and pushes the new value to the
 * gateway via `config.set`. On RPC failure reverts both the preset and the atom
 * to the user's pre-click value (`prev`) and notifies.
 *
 * Revert semantics (two layers):
 * 1. `prev` is captured at call time (before the optimistic write) so a
 *    parallel session-info reply — which may overwrite the live atom — can't
 *    poison the revert to the wrong value.
 * 2. `generation` is captured at call time (before the RPC) and re-checked on
 *    the catch path via `latestGeneration()`. If a newer click has bumped the
 *    generation ref between submit and failure, this A→B race reverts to A's
 *    `prev` would silently clobber B's already-committed optimistic value —
 *    so we skip the revert and let B stand. Mirrors `usageRequestRef` in
 *    `app/command-center/index.tsx:197-216` and `resumeRequestRef` in
 *    `app/session/hooks/use-session-actions/index.ts:281-349`.
 *
 * Empty `model` / `provider` early-out: a click that lands before the
 * composer knows the active model would otherwise call `setModelPreset` with
 * a `::` key and pollute the preset store. The pill renders nothing in that
 * state, but a stray call (e.g. a fast-resolved event handler) should still
 * no-op. Per-row submenu items always carry their own `model` and `provider`,
 * so this guard only catches the active-model-not-loaded window.
 *
 * Cross-component clobber (active row only): the pill and the per-row
 * submenu each own an independent `useRef(0)`. A rapid in-component A→B
 * click is fully covered (the same ref gets bumped). Cross-component
 * clobber is bounded by the menu state — only one of (pill, submenu on
 * the active row) is open at a time, and the submenu for a non-active
 * row is `isActive: false` and bails before reaching the RPC + revert.
 * If a future caller opens two writers to the same preset without this
 * invariant, switch to a shared module-level counter.
 */
export async function applyReasoningPatch({
  failMessage,
  generation,
  isActive,
  latestGeneration,
  model,
  next,
  prev,
  provider,
  request,
  sessionId
}: {
  failMessage: string
  /**
   * The caller's monotonically-increasing token. Captured at click time;
   * re-checked on the RPC failure path via `latestGeneration()`. If the
   * caller has bumped the token since this call started, this revert is
   * stale and must be skipped to avoid clobbering a newer optimistic value.
   */
  generation: number
  /**
   * Whether this write targets the user's currently-active model. The per-row
   * model picker submenu passes `false` for non-active rows (writes only the
   * preset, leaves the live session alone). The composer reasoning pill always
   * passes `true` (the composer pill only renders for the active model).
   */
  isActive: boolean
  /**
   * Accessor returning the caller's current generation token. Compared
   * against the `generation` captured on entry to decide whether this
   * call's revert is still the user's intent.
   */
  latestGeneration: () => number
  model: string
  next: string
  prev: string
  provider: string
  /**
   * JSON-RPC function — accepts the same shape as `useGatewayRequest`'s
   * `requestGateway`. `null` skips the RPC (preset + optimistic store are
   * the whole effect); pass `null` only when no live session is reachable.
   */
  request: (<T>(method: string, params?: Record<string, unknown>) => Promise<T>) | null
  /** Live session id. Required when `isActive && request` is truthy. */
  sessionId: string | null
}): Promise<void> {
  // Belt-and-suspenders: an empty model or provider would otherwise write a
  // `::` key into the preset store (model/provider lookup is keyed on both).
  // The pill is supposed to render nothing in this state, but a fast-resolved
  // event handler could still race the model atom.
  if (!model.trim() || !provider.trim()) {
    return
  }

  setModelPreset(provider, model, { effort: next })

  if (!isActive) {
    return
  }

  setCurrentReasoningEffort(next)

  if (!sessionId || !request) {
    return
  }

  try {
    await request('config.set', { key: 'reasoning', session_id: sessionId, value: next })
  } catch (err) {
    // A→B race guard: if the caller has bumped their generation ref since
    // this call entered, a newer write has committed. Reverting to `prev`
    // would clobber B back to A's pre-click value — drop the revert and let
    // B stand. The newer click's RPC failure handler will deal with its own
    // failure independently.
    if (latestGeneration() !== generation) {
      return
    }

    setCurrentReasoningEffort(prev)
    setModelPreset(provider, model, { effort: prev })
    notifyError(err, failMessage)
  }
}
