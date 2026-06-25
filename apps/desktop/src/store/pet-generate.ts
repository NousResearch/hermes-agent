import { atom } from 'nanostores'

import { persistString, storedString } from '@/lib/storage'
import { $gateway } from '@/store/gateway'
import { dispatchNativeNotification } from '@/store/native-notifications'
import { notify } from '@/store/notifications'
import { type PetInfo } from '@/store/pet'
import { applyAdoptedPet, type GatewayRequest } from '@/store/pet-gallery'
/**
 * Feature store for the "generate a pet" flow (Cmd-K → Pets → Generate).
 *
 * Three backend steps, mirrored as state here:
 *  - `pet.generate` produces N cheap base-look *drafts* keyed by a `token`.
 *  - `pet.hatch` turns the chosen draft into a full animated pet — installed but
 *    NOT active — and returns its renderer payload so we can preview all frames.
 *  - the user then *adopts* (`pet.select`) or *discards* (`pet.remove`) it.
 *
 * The store owns the draft set, the selected variant, the hatched preview, and
 * the busy/error status so the page is a thin view. Retry == regenerate (new
 * token). Kept separate from `pet-gallery` because its lifecycle (ephemeral
 * drafts + an unadopted preview) is unrelated to the long-lived gallery cache.
 */

// Generation is many grounded image calls — far longer than the default 30s RPC
// timeout. Drafts fan out 4 base looks; hatch fans out ~8 animation rows. Even
// parallelized, a cold provider call is slow, so we give these calls real
// headroom (the bug was "request timed out: pet.generate" on the 30s default).
const GENERATE_TIMEOUT_MS = 240_000
const HATCH_TIMEOUT_MS = 420_000

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
export function cleanPetName(prompt: string): string {
  const words = prompt
    .replace(/[^\p{L}\p{N}\s-]/gu, ' ')
    .split(/\s+/)
    .filter(Boolean)

  const meaningful = words.filter(w => !NAME_STOPWORDS.has(w.toLowerCase()))
  const picked = (meaningful.length ? meaningful : words).slice(0, 3)

  const name = picked
    .map(w => w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ')
    .slice(0, 28)
    .trim()

  return name || 'Pet'
}

export interface PetDraft {
  index: number
  /** Downscaled PNG data URI preview from the gateway. */
  dataUri: string
}

export type PetGenStatus =
  | 'idle'
  | 'generating'
  | 'ready'
  | 'hatching'
  | 'preview'
  | 'adopting'
  | 'error'
  | 'stale'

/** Live hatch step for the egg screen — which row is being drawn, then compose/save. */
export interface PetHatchStage {
  phase: 'row' | 'compose' | 'save'
  state?: string
  done?: number
  total?: number
}

export const $petGenStatus = atom<PetGenStatus>('idle')
export const $petGenStage = atom<PetHatchStage | null>(null)
export const $petGenError = atom<string | null>(null)

// Whether a reference-capable image backend is configured. `null` = not yet
// probed (treat as available so the prompt shows optimistically); the overlay
// re-probes on open and on return from settings.
export const $petGenAvailable = atom<boolean | null>(null)

/** A reference-capable image backend the user can pick for generation. */
export interface PetGenProvider {
  name: string
  label: string
  /** Whether this is the backend's default pick (no override needed). */
  default: boolean
}

const PROVIDER_KEY = 'hermes.desktop.petgen.provider'

/** Reference-capable providers available to pick (from `pet.generate.status`). */
export const $petGenProviders = atom<PetGenProvider[]>([])
/** The picked provider name; `''` means "use the backend default". Persisted. */
export const $petGenProvider = atom(storedString(PROVIDER_KEY) ?? '')

/** Set (and persist) the pet-gen provider override. `''` clears it. */
export function setPetGenProvider(name: string): void {
  $petGenProvider.set(name)
  persistString(PROVIDER_KEY, name || null)
}

/** Probe whether generation is possible (a reference-capable backend exists). */
export async function checkPetGenAvailable(request: GatewayRequest): Promise<void> {
  try {
    const res = await request<{ available: boolean; providers?: PetGenProvider[] }>('pet.generate.status')
    $petGenAvailable.set(Boolean(res?.available))
    const providers = res?.providers ?? []
    $petGenProviders.set(providers)
    // Drop a stale pick if that backend is no longer configured.
    const picked = $petGenProvider.get()

    if (picked && !providers.some(p => p.name === picked)) {
      setPetGenProvider('')
    }
  } catch {
    // Unknown (old backend / transient) — don't gate the UI on a failed probe.
    $petGenAvailable.set(true)
  }
}

/** Whether the dedicated "Generate a pet" Pokédex overlay is open. */
export const $petGenerateOpen = atom(false)

export function openPetGenerate(): void {
  // Resume an in-flight or finished-but-unadopted run (so a Stop-free close, or
  // a "done" notification click, lands back on the right step); only start on a
  // clean slate when nothing is going on.
  if ($petGenStatus.get() === 'idle') {
    resetPetGen()
  }

  $petGenerateOpen.set(true)
}

export function closePetGenerate(): void {
  $petGenerateOpen.set(false)
}

export const $petGenToken = atom<string | null>(null)
/** Prompt that produced the current draft token; hatch uses this for consistency. */
export const $petGenPrompt = atom<string>('')
export const $petGenDrafts = atom<PetDraft[]>([])
export const $petGenSelected = atom<number | null>(null)
/** The hatched-but-unadopted pet: its renderer payload, played in the preview. */
export const $petGenPreview = atom<PetInfo | null>(null)

// Live composer inputs live in atoms (not component state) so closing the
// overlay mid-flow — or letting it run in the background — and reopening (or
// clicking the "done" notification) restores exactly what you had.
export const $petGenInput = atom('')
export const $petGenRefImage = atom<string | null>(null)
export const $petGenRefName = atom('')

function isMissingMethod(error: unknown): boolean {
  const message = error instanceof Error ? error.message : String(error)

  return /method not found|-32601|unknown method|no such method/i.test(message)
}

/** Clear all generation state (before a fresh run). */
export function resetPetGen(): void {
  $petGenStatus.set('idle')
  $petGenStage.set(null)
  $petGenError.set(null)
  $petGenToken.set(null)
  $petGenPrompt.set('')
  $petGenDrafts.set([])
  $petGenSelected.set(null)
  $petGenPreview.set(null)
  $petGenInput.set('')
  $petGenRefImage.set(null)
  $petGenRefName.set('')
}

/**
 * Close-time cleanup: if a pet is already hatched but not adopted, discard it so
 * abandoned previews do not accumulate on disk. In-flight generate/hatch runs
 * are intentionally left alone (background-resumable).
 */
export function cleanupPetGenOnClose(request: GatewayRequest): void {
  const status = $petGenStatus.get()
  const preview = $petGenPreview.get()

  if ((status === 'preview' || status === 'adopting') && preview?.slug) {
    void request('pet.remove', { slug: preview.slug }).catch(() => {})
    resetPetGen()
  }
}

// A finished background run (overlay closed) nudges the user back: an in-app
// toast with a View action always, plus an OS notification when enabled and the
// app is in the background. Clicking either reopens the overlay to its state.
function notifyPetGenDone(title: string, message: string, kind: 'error' | 'success'): void {
  if ($petGenerateOpen.get()) {
    return
  }

  notify({ kind, title, message, action: { label: 'View', onClick: openPetGenerate } })
  // Pet generation isn't tied to a chat session — mark it global so the OS
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
  $petGenError.set(null)

  const drafts = $petGenDrafts.get()

  if (drafts.length > 0) {
    if ($petGenSelected.get() === null) {
      $petGenSelected.set(drafts[0]?.index ?? 0)
    }

    $petGenStatus.set('ready')

    return
  }

  $petGenStatus.set('idle')
  $petGenDrafts.set([])
  $petGenSelected.set(null)
  $petGenToken.set(null)
}

/**
 * Abandon the current drafts and return to the prompt (step 1). Stops any
 * in-flight generation; keeps the prompt text so the user can tweak + retry.
 */
export function discardDrafts(): void {
  gen.stop()
  $petGenDrafts.set([])
  $petGenSelected.set(null)
  $petGenToken.set(null)
  $petGenError.set(null)
  $petGenStatus.set('idle')
}

const hatch = cancelableRun()

// A Stop invalidates the in-flight hatch and drops back to the draft picker (the
// server still finishes, so we delete the pet it created).
/** Stop the in-flight hatch and return to the draft picker. */
export function cancelHatch(): void {
  hatch.stop()
  $petGenStage.set(null)
  $petGenError.set(null)
  $petGenStatus.set($petGenDrafts.get().length > 0 ? 'ready' : 'idle')
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
    const token = $petGenToken.get()

    if (token) {
      void request('pet.cancel', { token }).catch(() => {})
    }
  })

  // Starting a fresh generation round supersedes any unadopted preview pet.
  const preview = $petGenPreview.get()

  if (preview?.slug) {
    await request('pet.remove', { slug: preview.slug }).catch(() => {})
  }

  $petGenStatus.set('generating')
  $petGenError.set(null)
  $petGenPreview.set(null)
  $petGenDrafts.set([])
  $petGenSelected.set(null)

  // Stream drafts in as the backend finishes each one (pet.generate.progress),
  // so the grid fills live instead of sitting on placeholders until all N land.
  const off =
    $gateway.get()?.on<PetDraft & { token: string; count: number }>('pet.generate.progress', event => {
      const draft = event.payload

      // Token-only init event (no draft yet): learn the token immediately so an
      // early Stop can still tell the backend to cancel this run.
      if (draft?.token && !draft.dataUri) {
        if (gen.isCurrent(runId) && $petGenStatus.get() === 'generating') {
          $petGenToken.set(draft.token)
        }

        return
      }

      if (!draft?.dataUri || typeof draft.index !== 'number') {
        return
      }

      // Ignore events from a superseded/stopped run, and only stream while live.
      if (!gen.isCurrent(runId) || $petGenStatus.get() !== 'generating') {
        return
      }

      // Capture the token from the stream so a Stop can still hatch the partial set.
      if (draft.token) {
        $petGenToken.set(draft.token)
      }

      const current = $petGenDrafts.get()

      if (current.some(d => d.index === draft.index)) {
        return
      }

      $petGenDrafts.set(
        [...current, { index: draft.index, dataUri: draft.dataUri }].sort((a, b) => a.index - b.index)
      )
    }) ?? (() => {})

  try {
    const result = await request<{ ok: boolean; token: string; drafts: PetDraft[] }>(
      'pet.generate',
      {
        prompt,
        style: options.style ?? 'auto',
        count: options.count ?? 4,
        ...(referenceImage ? { referenceImage } : {}),
        ...($petGenProvider.get() ? { provider: $petGenProvider.get() } : {})
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

    $petGenToken.set(result.token)
    // Keep a concept for the hatch row prompts even on an image-only generate.
    $petGenPrompt.set(prompt || 'a custom pet')
    $petGenDrafts.set(result.drafts)
    $petGenSelected.set(result.drafts[0]?.index ?? 0)
    $petGenStatus.set('ready')
    notifyPetGenDone('Pet drafts ready', 'Your pet looks finished — pick one to hatch.', 'success')

    return true
  } catch (e) {
    if (!gen.isCurrent(runId)) {
      return false
    }

    if (isMissingMethod(e)) {
      $petGenStatus.set('stale')
    } else {
      $petGenStatus.set('error')
      $petGenError.set(e instanceof Error ? e.message : 'Could not generate pet drafts.')
      notifyPetGenDone('Pet generation failed', 'Reopen to try again.', 'error')
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
 * Hatch the selected draft into a full pet (installed but NOT yet active) and
 * load its renderer payload into the preview. Adoption is a separate, explicit
 * step (`adoptHatched`) so the user sees every frame play before committing.
 * Returns true when the preview is ready.
 */
export async function hatchSelected(request: GatewayRequest, options: HatchOptions): Promise<boolean> {
  const token = $petGenToken.get()
  const index = $petGenSelected.get()
  const name = options.name.trim()
  const concept = ($petGenPrompt.get() || options.prompt || name).trim()

  if (token === null || index === null || !name) {
    return false
  }

  // Hatch cancellation rides its own token (not the draft token): hatching
  // mid-generation leaves pet.generate releasing that token, which would race
  // the arm. The draft token still locates the staged image server-side.
  const cancelToken = crypto.randomUUID()
  const hatchRunId = hatch.begin()
  const controller = new AbortController()
  hatch.arm(() => {
    controller.abort()
    void request('pet.cancel', { token: cancelToken }).catch(() => {})
  })

  $petGenStatus.set('hatching')
  $petGenStage.set(null)
  $petGenError.set(null)

  // Stream the hatch steps (which row is drawing, then compose/save) to the egg
  // screen so a multi-minute hatch shows live progress instead of a black box.
  const offProgress =
    $gateway
      .get()
      ?.on<{ event: string; state?: string; done?: string; total?: string }>('pet.hatch.progress', event => {
        const p = event.payload

        if (!p || !hatch.isCurrent(hatchRunId) || $petGenStatus.get() !== 'hatching') {
          return
        }

        if (p.event === 'row' && p.state) {
          $petGenStage.set({
            phase: 'row',
            state: p.state,
            done: Number(p.done) || undefined,
            total: Number(p.total) || undefined
          })
        } else if (p.event === 'compose') {
          $petGenStage.set({ phase: 'compose' })
        } else if (p.event === 'save') {
          $petGenStage.set({ phase: 'save' })
        }
      }) ?? (() => {})

  try {
    const result = await request<{ ok: boolean; slug: string; displayName: string; pet?: PetInfo }>(
      'pet.hatch',
      {
        token,
        cancelToken,
        index,
        name,
        description: options.description ?? '',
        prompt: concept,
        style: options.style ?? 'auto',
        ...($petGenProvider.get() ? { provider: $petGenProvider.get() } : {})
      },
      HATCH_TIMEOUT_MS,
      controller.signal
    )

    // Stopped mid-hatch: the server created the pet anyway, so delete it.
    if (!hatch.isCurrent(hatchRunId)) {
      if (result?.slug) {
        void request('pet.remove', { slug: result.slug }).catch(() => {})
      }

      return false
    }

    if (!result?.ok || !result.pet?.spritesheetBase64) {
      throw new Error('hatch produced no preview')
    }

    $petGenPreview.set({ ...result.pet, enabled: true })
    $petGenStatus.set('preview')
    notifyPetGenDone('Your pet hatched', 'Reopen to name and adopt it.', 'success')

    return true
  } catch (e) {
    if (!hatch.isCurrent(hatchRunId)) {
      return false
    }

    $petGenStatus.set('error')
    $petGenError.set(e instanceof Error ? e.message : 'Could not hatch the pet.')
    notifyPetGenDone('Hatching failed', 'Reopen to try again.', 'error')

    return false
  } finally {
    offProgress()

    if (hatch.isCurrent(hatchRunId)) {
      $petGenStage.set(null)
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
 * Adopt the previewed pet: optionally rename it to the user's chosen name (set
 * on the reveal screen), activate it (`pet.select`), refresh the gallery + live
 * mascot, and clear generation state. No-op unless a preview exists.
 */
export async function adoptHatched(request: GatewayRequest, name?: string): Promise<AdoptOutcome> {
  const preview = $petGenPreview.get()

  if (!preview?.slug) {
    return { ok: false }
  }

  $petGenStatus.set('adopting')
  $petGenError.set(null)

  try {
    // Name is collected after hatch, so apply it before activating. The rename
    // also realigns the slug to the chosen name (so lists show what the user
    // typed, not the prompt), so adopt the *returned* slug. Best-effort: a
    // rename failure shouldn't block adopting under the provisional slug.
    const finalName = name?.trim()
    let adoptSlug = preview.slug

    if (finalName && finalName !== preview.displayName) {
      const renamed = await request<{ ok: boolean; slug: string }>('pet.rename', {
        slug: preview.slug,
        name: finalName
      }).catch(() => null)

      if (renamed?.slug) {
        adoptSlug = renamed.slug
      }
    }

    const result = await request<{ ok: boolean; slug: string; displayName: string }>('pet.select', {
      slug: adoptSlug
    })

    if (!result?.ok) {
      throw new Error('adopt failed')
    }

    // pet.select already set the active mascot (disk + config). Reflect it
    // locally — no remote petdex manifest fetch — and close immediately.
    resetPetGen()
    void applyAdoptedPet(request, result.slug, result.displayName)

    return { ok: true, slug: result.slug, displayName: result.displayName }
  } catch (e) {
    $petGenStatus.set('preview')
    $petGenError.set(e instanceof Error ? e.message : 'Could not adopt the pet.')

    return { ok: false }
  }
}

/**
 * Throw away the previewed pet (`pet.remove`) and return to the draft picker so
 * the user can choose another base or regenerate. Best-effort on the delete.
 */
export async function discardHatched(request: GatewayRequest): Promise<void> {
  const preview = $petGenPreview.get()

  if (preview?.slug) {
    await request('pet.remove', { slug: preview.slug }).catch(() => {})
  }

  $petGenPreview.set(null)
  $petGenError.set(null)
  $petGenStatus.set($petGenDrafts.get().length > 0 ? 'ready' : 'idle')
}
