import { useStore } from '@nanostores/react'
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { SETTINGS_ROUTE } from '@/app/routes'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
import { DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { GenerateButton } from '@/components/ui/generate-button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Egg, ImageIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $petGenAvailable,
  $petGenDrafts,
  $petGenError,
  $petGenInput,
  $petGenPreview,
  $petGenRefImage,
  $petGenRefName,
  $petGenRemixConfirmed,
  $petGenSelected,
  $petGenStage,
  $petGenStatus,
  adoptHatched,
  cancelGenerate,
  checkPetGenAvailable,
  cleanPetName,
  closePetGenerate,
  discardDrafts,
  discardHatched,
  generateDrafts,
  hatchSelected,
  markRemixConfirmed
} from '@/store/pet-generate'

import { DraftGrid } from './components/draft-grid'
import { EmptyHint } from './components/empty-hint'
import { GenerateUnavailable } from './components/generate-unavailable'
import { HatchPreview } from './components/hatch-preview'
import { HatchingView } from './components/hatching-view'
import { ProviderPicker } from './components/provider-picker'
import { ReferenceChip } from './components/reference-chip'
import { readReferenceImage } from './lib/read-reference-image'

// The generate → hatch → adopt controller. A thin view over the `pet-generate`
// store; the store owns the steps and persists inputs across close/reopen.
export function PetGenerateContent() {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet
  const { requestGateway } = useGatewayRequest()
  const navigate = useNavigate()

  const status = useStore($petGenStatus)
  const error = useStore($petGenError)
  const available = useStore($petGenAvailable)
  // `null` = not yet probed → stay optimistic (show the prompt); only the
  // confirmed-no-backend case swaps in the setup card.
  const unavailable = available === false
  const drafts = useStore($petGenDrafts)
  const selected = useStore($petGenSelected)
  const preview = useStore($petGenPreview)
  const stage = useStore($petGenStage)

  // Inputs live in atoms so they survive a close/reopen (and background runs).
  const prompt = useStore($petGenInput)
  const refImage = useStore($petGenRefImage)
  const refName = useStore($petGenRefName)
  const fileRef = useRef<HTMLInputElement>(null)

  // The draft awaiting the one-time "remix regenerates" confirmation.
  const [remixPending, setRemixPending] = useState<{ dataUri: string } | null>(null)

  // Probe backend availability on open — and again whenever the content
  // remounts (e.g. after returning from the providers settings), so adding a
  // key flips the setup card to the prompt with no manual refresh.
  useEffect(() => {
    void checkPetGenAvailable(requestGateway)
  }, [requestGateway])

  const busy = status === 'generating' || status === 'hatching'
  const hasDrafts = drafts.length > 0
  const generating = status === 'generating'

  // The idle "describe a pet" state — egg + suggestions get generous, equidistant
  // breathing room (gap-4) from the prompt; the working states stay compact.
  const isEmptyState =
    !hasDrafts &&
    !generating &&
    status !== 'hatching' &&
    status !== 'preview' &&
    status !== 'adopting' &&
    status !== 'stale'

  const generate = () => {
    if ((prompt.trim() || refImage) && !busy) {
      void generateDrafts(requestGateway, { prompt: prompt.trim(), referenceImage: refImage ?? undefined })
    }
  }

  const clearReference = () => {
    $petGenRefImage.set(null)
    $petGenRefName.set('')
  }

  const pickReference = (file: File | undefined) => {
    if (!file) {
      return
    }

    const mapReferenceError = (reason: unknown): string => {
      const message = reason instanceof Error ? reason.message.toLowerCase() : ''

      return message.includes('too large') ? copy.referenceImageTooLarge : copy.referenceImageInvalid
    }

    void readReferenceImage(file)
      .then(dataUrl => {
        $petGenRefImage.set(dataUrl)
        $petGenRefName.set(file.name)
        // Clear picker-only errors once the reference is valid again.

        if ($petGenStatus.get() === 'error' && $petGenDrafts.get().length === 0) {
          $petGenStatus.set('idle')
          $petGenError.set(null)
        }
      })
      .catch(reason => {
        $petGenRefImage.set(null)
        $petGenRefName.set('')
        $petGenError.set(mapReferenceError(reason))

        if (!busy) {
          $petGenStatus.set('error')
        }
      })
  }

  // One-click an example prompt straight into a draft round.
  const runExample = (example: string) => {
    $petGenInput.set(example)
    void generateDrafts(requestGateway, { prompt: example })
  }

  // A remix re-runs generation grounded on an existing draft — same prompt, stay
  // on step 2 — so the user explores variations without starting over.
  const runRemix = (draft: { dataUri: string }) => {
    void generateDrafts(requestGateway, { prompt: prompt.trim(), referenceImage: draft.dataUri })
  }

  // Slow, and it replaces the current drafts — so confirm once, then remember it.
  const remixDraft = (draft: { dataUri: string }) => {
    if (busy) {
      return
    }

    if ($petGenRemixConfirmed.get()) {
      runRemix(draft)

      return
    }

    setRemixPending(draft)
  }

  // Hatch the selected draft. The user can pick one before the rest stream in —
  // if so, abort the remaining generations first (keeping the drafts we have).
  // The prompt is grounding text, not a label; the user names it on reveal.
  const hatch = () => {
    if (selected === null) {
      return
    }

    if (generating) {
      cancelGenerate()
    }

    void hatchSelected(requestGateway, { name: cleanPetName(prompt), prompt: prompt.trim() })
  }

  const adopt = (finalName: string) => {
    void adoptHatched(requestGateway, finalName).then(out => {
      if (out.ok) {
        triggerHaptic('crisp')
        closePetGenerate()
      }
    })
  }

  // The header title tracks the phase instead of sticking on "Generate a pet".
  const headerTitle =
    status === 'hatching' ? copy.spawning : status === 'preview' || status === 'adopting' ? copy.hatched : copy.title

  // Send the user to set up a key without closing — the overlay yields to the
  // settings route (useRouteOverlayActive) and reappears + re-checks on return.
  const setupImageGen = () => navigate(`${SETTINGS_ROUTE}?tab=providers`)

  // Prompt input only belongs on the describe/draft screens (and never when
  // there's no backend to generate with).
  const showPrompt = !unavailable && status !== 'hatching' && status !== 'preview' && status !== 'adopting'

  return (
    <>
      {unavailable ? (
        <DialogTitle className="sr-only">{copy.title}</DialogTitle>
      ) : (
        <DialogHeader>
          <DialogTitle icon={Egg}>{headerTitle}</DialogTitle>
        </DialogHeader>
      )}

      <div className={cn('flex min-h-0 flex-1 flex-col', isEmptyState ? 'gap-4' : 'gap-2.5')}>
        {/* Concept prompt with the inline sparkle generate/stop affordance (the
            same primitive as the commit-message + project-idea fields). */}
        {showPrompt && (
          <div className="flex flex-col gap-1.5">
            <div className="relative">
              <Input
                autoFocus
                className="pr-9"
                onChange={event => $petGenInput.set(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') {
                    event.preventDefault()
                    generate()
                  }
                }}
                placeholder={copy.placeholder}
                value={prompt}
              />
              <GenerateButton
                className="absolute right-1 top-1/2 -translate-y-1/2"
                disabled={!prompt.trim() && !refImage}
                generating={generating}
                generatingLabel={t.common.cancel}
                label={copy.generate}
                // Inline cancel should match step-2 cancel semantics: abort and
                // return to step 1 (prompt retained for quick tweaks).
                onCancel={discardDrafts}
                onGenerate={generate}
              />
            </div>

            <div className="flex items-center gap-2">
              <ProviderPicker />
              {refImage ? (
                <ReferenceChip name={refName} onRemove={clearReference} src={refImage} />
              ) : (
                <button
                  className="ml-auto flex h-6 items-center gap-1.5 text-[0.6875rem] text-(--ui-text-tertiary) transition hover:text-foreground"
                  onClick={() => fileRef.current?.click()}
                  type="button"
                >
                  <ImageIcon className="size-3" />
                  Add a reference
                </button>
              )}
            </div>

            {/* Optional reference photo — make a pet from the user's own image.
                Styled like the chat composer's attachment pill. */}
            <Input
              accept="image/*"
              className="hidden"
              onChange={event => {
                pickReference(event.target.files?.[0])
                event.target.value = ''
              }}
              ref={fileRef}
              type="file"
            />
          </div>
        )}

        {/* Hatch failed but the drafts are still here — show why above the grid so
            the user can re-pick and retry without losing their options. */}
        {status === 'error' && hasDrafts && (
          <Alert variant="destructive">
            <AlertDescription>{error || copy.genericError}</AlertDescription>
          </Alert>
        )}

        {unavailable ? (
          <GenerateUnavailable onSetup={setupImageGen} />
        ) : status === 'stale' ? (
          <Alert variant="destructive">
            <AlertDescription>{copy.staleBackend}</AlertDescription>
          </Alert>
        ) : status === 'hatching' ? (
          <HatchingView stage={stage} />
        ) : (status === 'preview' || status === 'adopting') && preview ? (
          <HatchPreview
            adopting={status === 'adopting'}
            error={error}
            onAdopt={adopt}
            onDiscard={() => void discardHatched(requestGateway)}
            pet={preview}
          />
        ) : !hasDrafts && !generating ? (
          // Doubles as the error-empty state — the failure reason rides the
          // dialog's footer banner, so here we just offer the retry sparks.
          <EmptyHint onExample={runExample} />
        ) : (
          <DraftGrid
            drafts={drafts}
            generating={generating}
            hasDrafts={hasDrafts}
            onCancel={discardDrafts}
            onHatch={hatch}
            onRemix={remixDraft}
            onSelect={index => $petGenSelected.set(index)}
            selected={selected}
          />
        )}
      </div>

      <ConfirmDialog
        confirmLabel={copy.remix}
        description={copy.remixConfirmBody}
        onClose={() => setRemixPending(null)}
        onConfirm={() => {
          markRemixConfirmed()

          if (remixPending) {
            runRemix(remixPending)
          }
        }}
        open={remixPending !== null}
        title={copy.remixConfirmTitle}
      />
    </>
  )
}
