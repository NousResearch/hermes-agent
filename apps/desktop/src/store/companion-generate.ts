import { randomUuid } from '@hermes/shared'
import { atom } from 'nanostores'

import { isMissingRpcMethod } from '@/lib/gateway-rpc'
import { persistBoolean, persistString, storedBoolean, storedString } from '@/lib/storage'
import { capitalize } from '@/lib/text'
import { type CompanionInfo } from '@/store/companion'
import { applyAdoptedCompanion, type GatewayRequest } from '@/store/companion-gallery'
import { $gateway } from '@/store/gateway'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
/**
 * Feature store for the "generate a companion" flow (Cmd-K → Companions → Generate).
 *
 * Three backend steps, mirrored as state here:
 *  - `companion.generate` produces N cheap base-look *drafts* keyed by a `token`.
 *  - `companion.hatch` turns the chosen draft into a full animated companion — installed but
 *    NOT active — and returns its renderer payload so we can preview all frames.
 *  - the user then *adopts* (`companion.select`) or *discards* (`companion.remove`) it.
 *
 * The store owns the draft set, the selected variant, the hatched preview, and
 * the busy/error status so the page is a thin view. Retry == regenerate (new
 * token). Kept separate from `companion-gallery` because its lifecycle (ephemeral
 * drafts + an unadopted preview) is unrelated to the long-lived gallery cache.
 */

// Generation is many grounded image calls — far longer than the default 30s RPC
// timeout. Drafts fan out 4 base looks; hatch fans out ~8 animation rows. The
// quality-first default (OpenAI image via OpenRouter) is slow, and each hatch
// row can retry up to 3x (300s/call) across 2 parallel waves, so the absolute
// backend worst case is ~30 min. The hatch ceiling sits above that (1h) so the
// frontend never throws "request timed out" before the backend has actually
// exhausted its own retries — the background-resumable notify path is the real
// UX safety net (the user can close the modal and get pinged on completion).
const GENERATE_TIMEOUT_MS = 420_000
const HATCH_TIMEOUT_MS = 3_600_000

// Filler words to drop when deriving a default name from a free-text prompt.
const NAME_STOPWORDS = new Set([
  'a',
  'an',
  'and',
  'at',
  'by',
  'cute',
  'for',
  'from',
  'in',
  'of',
  'on',
  'style',
  'the',
  'to',
  'with'
])

/**
 * Derive a short, friendly default name from a generation prompt. The prompt
 * (e.g. "2d dragon in the style of ragnarok online") is grounding text, not a
 * name — using it verbatim makes a terrible label + slug. We keep the first few
 * meaningful words, title-cased and capped, so a blank adopt still reads well.
 * The user can always override on the reveal screen or rename later.
 */
export function cleanCompanionName(prompt: string): string {
  const words = prompt
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)

  const meaningful = words.filter(w => !NAME_STOPWORDS.has(w.toLowerCase()))
  const picked = (meaningful.length ? meaningful : words).slice(0, 3)

  const name = picked.map(capitalize).join(' ').slice(0, 28).trim()

  return name || 'Companion'
}

export interface CompanionDraft {
  index: number
  /** Downscaled PNG data URI preview from the gateway. */
  dataUri: string
}

export type CompanionGenStatus = 'idle' | 'generating' | 'ready' | 'hatching' | 'preview' | 'adopting' | 'error' | 'stale'

/** Live hatch step for the egg screen — which row is being drawn, then compose/save. */
export interface CompanionHatchStage {
  phase: 'row' | 'compose' | 'save'
  state?: string
  done?: number
  total?: number
}

export const $companionGenStatus = atom<CompanionGenStatus>('idle')
export const $companionGenStage = atom<CompanionHatchStage | null>(null)
export const $companionGenError = atom<string | null>(null)

// Whether a reference-capable image backend is configured. `null` = not yet
// probed (treat as available so the prompt shows optimistically); the overlay
// re-probes on open and on return from settings.
export const $companionGenAvailable = atom<boolean | null>(null)

/** A reference-capable image backend the user can pick for generation. */
export interface CompanionGenProvider {
  name: string
  label: string
  /** Whether this is the backend's default pick (no override needed). */
  default: boolean
}

const PROVIDER_KEY = 'hermes.desktop.companiongen.provider'
const REMIX_CONFIRMED_KEY = 'hermes.desktop.companiongen.remixConfirmed'

/** Reference-capable providers available to pick (from `companion.generate.status`). */
export const $companionGenProviders = atom<CompanionGenProvider[]>([])
/** The picked provider name; `''` means "use the backend default". Persisted. */
export const $companionGenProvider = atom(storedString(PROVIDER_KEY) ?? '')

/** Set (and persist) the companion-gen provider override. `''` clears it. */
export function setCompanionGenProvider(name: string): void {
  $companionGenProvider.set(name)
  persistString(PROVIDER_KEY, name || null)
}

/** Whether the user has acknowledged the one-time "remix regenerates" notice. */
export const $companionGenRemixConfirmed = atom(storedBoolean(REMIX_CONFIRMED_KEY, false))

/** Remember that the remix notice has been shown so we don't ask again. */
export function markRemixConfirmed(): void {
  $companionGenRemixConfirmed.set(true)
  persistBoolean(REMIX_CONFIRMED_KEY, true)
}

/** Probe whether generation is possible (a reference-capable backend exists). */
export async function checkCompanionGenAvailable(request: GatewayRequest): Promise<void> {
  try {
    const res = await request<{ available: boolean; providers?: CompanionGenProvider[] }>('companion.generate.status')
    $companionGenAvailable.set(Boolean(res?.available))
    const providers = res?.providers ?? []
    $companionGenProviders.set(providers)
    // Drop a stale pick if that backend is no longer configured.
    const picked = $companionGenProvider.get()

    if (picked && !providers.some(p => p.name === picked)) {
      setCompanionGenProvider('')
    }
  } catch {
    // Unknown (old backend / transient) — don't gate the UI on a failed probe.
    $companionGenAvailable.set(true)
  }
}

/** Whether the dedicated "Generate a companion" Pokédex overlay is open. */
export const $companionGenerateOpen = atom(false)

export function openCompanionGenerate(): void {
  // Resume an in-flight or finished-but-unadopted run (so a Stop-free close, or
  // a "done" notification click, lands back on the right step); only start on a
  // clean slate when nothing is going on.
  if ($companionGenStatus.get() === 'idle') {
    resetCompanionGen()
  }

  $companionGenerateOpen.set(true)
}

export function closeCompanionGenerate(): void {
  $companionGenerateOpen.set(false)
}

export const $companionGenToken = atom<string | null>(null)
/** Prompt that produced the current draft token; hatch uses this for consistency. */
export const $companionGenPrompt = atom<string>('')
export const $companionGenDrafts = atom<CompanionDraft[]>([])
export const $companionGenSelected = atom<number | null>(null)
/** The hatched-but-unadopted companion: its renderer payload, played in the preview. */
export const $companionGenPreview = atom<CompanionInfo | null>(null)

// Live composer inputs live in atoms (not component state) so closing the
// overlay mid-flow — or letting it run in the background — and reopening (or
// clicking the "done" notification) restores exactly what you had.
export const $companionGenInput = atom('')
export const $companionGenRefImage = atom<string | null>(null)
export const $companionGenRefName = atom('')

/** Clear all generation state (before a fresh run). */
export function resetCompanionGen(): void {
  $companionGenStatus.set('idle')
  $companionGenStage.set(null)
  $companionGenError.set(null)
  $companionGenToken.set(null)
  $companionGenPrompt.set('')
  $companionGenDrafts.set([])
  $companionGenSelected.set(null)
  $companionGenPreview.set(null)
  $companionGenInput.set('')
  $companionGenRefImage.set(null)
  $companionGenRefName.set('')
}

/**
 * Close-time cleanup: if a companion is already hatched but not adopted, discard it so
 * abandoned previews do not accumulate on disk. In-flight generate/hatch runs
 * are intentionally left alone (background-resumable).
 */
export function cleanupCompanionGenOnClose(request: GatewayRequest): void {
  const status = $companionGenStatus.get()
  const preview = $companionGenPreview.get()

  if ((status === 'preview' || status === 'adopting') && preview?.slug) {
    void request('companion.remove', { slug: preview.slug }).catch(() => {})
    resetCompanionGen()
  }
}

// A finished background run (overlay closed) nudges the user back: an in-app
// toast with a View action always, plus an OS notification when enabled and the
// app is in the background. Clicking either reopens the overlay to its state.
function notifyCompanionGenDone(title: string, message: string, kind: 'error' | 'success'): void {
  if ($companionGenerateOpen.get()) {
    return
  }

  notify({ kind, title, message, action: { label: 'View', onClick: openCompanionGenerate } })
  // Companion generation isn't tied to a chat session — mark it global so the OS
  // notification fires whenever the user is away, even with no active session
  // (the common case: generating from the command center with no conversation).
  dispatchNativeNotification({ kind: 'backgroundDone', title, body: message, global: true })
}

interface GenerateOptions {
  prompt: string
  style?: string
  count?: number
  /** Optional data-URL reference image — every draft is grounded on it. */
  referenceImage?: string
}

// A Stop (or a fresh round) must invalidate the in-flight call. This primitive
// pairs a monotonic run id with the current run's cancel fn; `begin` opens a
// run, `isCurrent` gates stale callbacks/events, `arm` registers the aborter,
// `stop` supersedes + fires it. Drives both the draft and hatch flows.
interface Run {
  begin: () => number
  isCurrent: (id: number) => boolean
  arm: (cancel: () => void) => void
  stop: () => void
  disarmIf: (id: number) => void
}

function cancelableRun(): Run {
  let id = 0
  let cancel: (() => void) | null = null

  return {
    begin: () => (id += 1),
    isCurrent: n => n === id,
    arm: fn => {
      cancel = fn
    },
    stop: () => {
      id += 1
      cancel?.()
      cancel = null
    },
    disarmIf: n => {
      if (n === id) {
        cancel = null
      }
    }
  }
}

const gen = cancelableRun()

/**
 * Stop the in-flight draft generation (real abort). If any drafts have already
 * streamed in, keep them and drop into the ready/picker state (no reason to wait
 * for all 4) — otherwise reset to idle.
 */
export function cancelGenerate(): void {
  gen.stop()
  $companionGenError.set(null)

  const drafts = $companionGenDrafts.get()

  if (drafts.length > 0) {
    if ($companionGenSelected.get() === null) {
      $companionGenSelected.set(drafts[0]?.index ?? 0)
    }

    $companionGenStatus.set('ready')

    return
  }

  $companionGenStatus.set('idle')
  $companionGenDrafts.set([])
  $companionGenSelected.set(null)
  $companionGenToken.set(null)
}

/**
 * Abandon the current drafts and return to the prompt (step 1). Stops any
 * in-flight generation; keeps the prompt text so the user can tweak + retry.
 */
export function discardDrafts(): void {
  gen.stop()
  $companionGenDrafts.set([])
  $companionGenSelected.set(null)
  $companionGenToken.set(null)
  $companionGenError.set(null)
  $companionGenStatus.set('idle')
}

const hatch = cancelableRun()

// A Stop invalidates the in-flight hatch and drops back to the draft picker (the
// server still finishes, so we delete the companion it created).
/** Stop the in-flight hatch and return to the draft picker. */
export function cancelHatch(): void {
  hatch.stop()
  $companionGenStage.set(null)
  $companionGenError.set(null)
  $companionGenStatus.set($companionGenDrafts.get().length > 0 ? 'ready' : 'idle')
}

/** Generate (or retry) a fresh set of base-look drafts for `prompt`. */
export async function generateDrafts(request: GatewayRequest, options: GenerateOptions): Promise<boolean> {
  const prompt = options.prompt.trim()
  const referenceImage = options.referenceImage

  // Need *something* to ground on: a description or a reference image.
  if (!prompt && !referenceImage) {
    return false
  }

  const runId = gen.begin()
  const controller = new AbortController()
  gen.arm(() => {
    controller.abort()
    const token = $companionGenToken.get()

    if (token) {
      void request('companion.cancel', { token }).catch(() => {})
    }
  })

  // Starting a fresh generation round supersedes any unadopted preview companion.
  const preview = $companionGenPreview.get()

  if (preview?.slug) {
    await request('companion.remove', { slug: preview.slug }).catch(() => {})
  }

  $companionGenStatus.set('generating')
  $companionGenError.set(null)
  $companionGenPreview.set(null)
  $companionGenDrafts.set([])
  $companionGenSelected.set(null)

  // Stream drafts in as the backend finishes each one (companion.generate.progress),
  // so the grid fills live instead of sitting on placeholders until all N land.
  const off =
    $gateway.get()?.on('companion.generate.progress', event => {
      // Shared map types this payload as an open record; the companion backend's draft shape is desktop-owned.
      const draft = event.payload as (CompanionDraft & { count: number; token: string }) | undefined

      // Token-only init event (no draft yet): learn the token immediately so an
      // early Stop can still tell the backend to cancel this run.
      if (draft?.token && !draft.dataUri) {
        if (gen.isCurrent(runId) && $companionGenStatus.get() === 'generating') {
          $companionGenToken.set(draft.token)
        }

        return
      }

      if (!draft?.dataUri || typeof draft.index !== 'number') {
        return
      }

      // Ignore events from a superseded/stopped run, and only stream while live.
      if (!gen.isCurrent(runId) || $companionGenStatus.get() !== 'generating') {
        return
      }

      // Capture the token from the stream so a Stop can still hatch the partial set.
      if (draft.token) {
        $companionGenToken.set(draft.token)
      }

      const current = $companionGenDrafts.get()

      if (current.some(d => d.index === draft.index)) {
        return
      }

      $companionGenDrafts.set([...current, { index: draft.index, dataUri: draft.dataUri }].sort((a, b) => a.index - b.index))
    }) ?? (() => {})

  try {
    const result = await request<{ ok: boolean; token: string; drafts: CompanionDraft[] }>(
      'companion.generate',
      {
        prompt,
        style: options.style ?? 'auto',
        count: options.count ?? 4,
        ...(referenceImage ? { referenceImage } : {}),
        ...($companionGenProvider.get() ? { provider: $companionGenProvider.get() } : {})
      },
      GENERATE_TIMEOUT_MS,
      controller.signal
    )

    // Stopped (or superseded by a newer round) while the RPC was in flight.
    if (!gen.isCurrent(runId)) {
      return false
    }

    if (!result?.ok || !result.drafts?.length) {
      throw new Error('generation produced no drafts')
    }

    $companionGenToken.set(result.token)
    // Keep a concept for the hatch row prompts even on an image-only generate.
    $companionGenPrompt.set(prompt || 'a custom companion')
    $companionGenDrafts.set(result.drafts)
    $companionGenSelected.set(result.drafts[0]?.index ?? 0)
    $companionGenStatus.set('ready')
    notifyCompanionGenDone('Companion drafts ready', 'Your companion looks finished — pick one to hatch.', 'success')

    return true
  } catch (e) {
    if (!gen.isCurrent(runId)) {
      return false
    }

    if (isMissingRpcMethod(e)) {
      $companionGenStatus.set('stale')
    } else {
      $companionGenStatus.set('error')
      $companionGenError.set(e instanceof Error ? e.message : 'Could not generate companion drafts.')
      notifyCompanionGenDone('Companion generation failed', 'Reopen to try again.', 'error')
    }

    return false
  } finally {
    off()
    gen.disarmIf(runId)
  }
}

interface HatchOptions {
  name: string
  description?: string
  prompt?: string
  style?: string
}

/**
 * Hatch the selected draft into a full companion (installed but NOT yet active) and
 * load its renderer payload into the preview. Adoption is a separate, explicit
 * step (`adoptHatched`) so the user sees every frame play before committing.
 * Returns true when the preview is ready.
 */
export async function hatchSelected(request: GatewayRequest, options: HatchOptions): Promise<boolean> {
  const token = $companionGenToken.get()
  const index = $companionGenSelected.get()
  const name = options.name.trim()
  const concept = ($companionGenPrompt.get() || options.prompt || name).trim()

  if (token === null || index === null || !name) {
    return false
  }

  // Hatch cancellation rides its own token (not the draft token): hatching
  // mid-generation leaves companion.generate releasing that token, which would race
  // the arm. The draft token still locates the staged image server-side.
  const cancelToken = randomUuid()
  const hatchRunId = hatch.begin()
  const controller = new AbortController()
  hatch.arm(() => {
    controller.abort()
    void request('companion.cancel', { token: cancelToken }).catch(() => {})
  })

  $companionGenStatus.set('hatching')
  $companionGenStage.set(null)
  $companionGenError.set(null)

  // Stream the hatch steps (which row is drawing, then compose/save) to the egg
  // screen so a multi-minute hatch shows live progress instead of a black box.
  const offProgress =
    $gateway.get()?.on('companion.hatch.progress', event => {
      const p = event.payload as { done?: string; event: string; state?: string; total?: string } | undefined

      if (!p || !hatch.isCurrent(hatchRunId) || $companionGenStatus.get() !== 'hatching') {
        return
      }

      if (p.event === 'row' && p.state) {
        $companionGenStage.set({
          phase: 'row',
          state: p.state,
          done: Number(p.done) || undefined,
          total: Number(p.total) || undefined
        })
      } else if (p.event === 'compose') {
        $companionGenStage.set({ phase: 'compose' })
      } else if (p.event === 'save') {
        $companionGenStage.set({ phase: 'save' })
      }
    }) ?? (() => {})

  try {
    const result = await request<{ ok: boolean; slug: string; displayName: string; companion?: CompanionInfo }>(
      'companion.hatch',
      {
        token,
        cancelToken,
        index,
        name,
        description: options.description ?? '',
        prompt: concept,
        style: options.style ?? 'auto',
        ...($companionGenProvider.get() ? { provider: $companionGenProvider.get() } : {})
      },
      HATCH_TIMEOUT_MS,
      controller.signal
    )

    // Stopped mid-hatch: the server created the companion anyway, so delete it.
    if (!hatch.isCurrent(hatchRunId)) {
      if (result?.slug) {
        void request('companion.remove', { slug: result.slug }).catch(() => {})
      }

      return false
    }

    if (!result?.ok || !result.companion?.spritesheetBase64) {
      throw new Error('hatch produced no preview')
    }

    $companionGenPreview.set({ ...result.companion, enabled: true })
    $companionGenStatus.set('preview')
    notifyCompanionGenDone('Your companion hatched', 'Reopen to name and adopt it.', 'success')

    return true
  } catch (e) {
    if (!hatch.isCurrent(hatchRunId)) {
      return false
    }

    $companionGenStatus.set('error')
    $companionGenError.set(e instanceof Error ? e.message : 'Could not hatch the companion.')
    notifyCompanionGenDone('Hatching failed', 'Reopen to try again.', 'error')

    return false
  } finally {
    offProgress()

    if (hatch.isCurrent(hatchRunId)) {
      $companionGenStage.set(null)
      hatch.disarmIf(hatchRunId)
    }
  }
}

export interface AdoptOutcome {
  ok: boolean
  slug?: string
  displayName?: string
}

/**
 * Adopt the previewed companion: optionally rename it to the user's chosen name (set
 * on the reveal screen), activate it (`companion.select`), refresh the gallery + live
 * mascot, and clear generation state. No-op unless a preview exists.
 */
export async function adoptHatched(request: GatewayRequest, name?: string): Promise<AdoptOutcome> {
  const preview = $companionGenPreview.get()

  if (!preview?.slug) {
    return { ok: false }
  }

  $companionGenStatus.set('adopting')
  $companionGenError.set(null)

  try {
    // Name is collected after hatch, so apply it before activating. The rename
    // also realigns the slug to the chosen name (so lists show what the user
    // typed, not the prompt), so adopt the *returned* slug. Best-effort: a
    // rename failure shouldn't block adopting under the provisional slug.
    const finalName = name?.trim()
    let adoptSlug = preview.slug

    if (finalName && finalName !== preview.displayName) {
      const renamed = await request<{ ok: boolean; slug: string }>('companion.rename', {
        slug: preview.slug,
        name: finalName
      }).catch(() => null)

      if (renamed?.slug) {
        adoptSlug = renamed.slug
      }
    }

    const result = await request<{ ok: boolean; slug: string; displayName: string }>('companion.select', {
      slug: adoptSlug
    })

    if (!result?.ok) {
      throw new Error('adopt failed')
    }

    // companion.select already set the active mascot (disk + config). Reflect it
    // locally — no remote companiondex manifest fetch — and close immediately.
    resetCompanionGen()
    void applyAdoptedCompanion(request, result.slug, result.displayName)

    return { ok: true, slug: result.slug, displayName: result.displayName }
  } catch (e) {
    $companionGenStatus.set('preview')
    $companionGenError.set(e instanceof Error ? e.message : 'Could not adopt the companion.')

    return { ok: false }
  }
}

/**
 * Throw away the previewed companion (`companion.remove`) and return to the draft picker so
 * the user can choose another base or regenerate. Best-effort on the delete.
 */
export async function discardHatched(request: GatewayRequest): Promise<void> {
  const preview = $companionGenPreview.get()

  if (preview?.slug) {
    await request('companion.remove', { slug: preview.slug }).catch(() => {})
  }

  $companionGenPreview.set(null)
  $companionGenError.set(null)
  $companionGenStatus.set($companionGenDrafts.get().length > 0 ? 'ready' : 'idle')
}
