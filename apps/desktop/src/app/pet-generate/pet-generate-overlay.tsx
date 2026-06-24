/**
 * "Hatch a Pet" — a dedicated, Pokédex-style overlay for pet generation.
 *
 * Previously generation lived as a cramped nested page inside the Cmd-K command
 * palette (~34rem popover). This is its own full Radix dialog with room to
 * breathe: a device-framed header, its own concept prompt, a roomy draft grid
 * that streams in live, and the egg-hatch + reveal flow. It's a thin view over
 * the `pet-generate` store; the store owns the generate → hatch → adopt steps.
 */

import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { PetEggHatch } from '@/components/pet/pet-egg-hatch'
import { PetStarShower } from '@/components/pet/pet-star-shower'
import { PetSprite } from '@/components/pet/pet-sprite'
import { PixelEggSprite } from '@/components/pet/pixel-egg-sprite'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { GenerateButton } from '@/components/ui/generate-button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Egg, Loader2, PawPrint, RefreshCw } from '@/lib/icons'
import { selectableCardClass } from '@/lib/selectable-card'
import { cn } from '@/lib/utils'
import { type PetInfo } from '@/store/pet'
import {
  $petGenDrafts,
  $petGenerateOpen,
  $petGenError,
  $petGenPreview,
  $petGenSelected,
  $petGenStage,
  $petGenStatus,
  adoptHatched,
  cancelGenerate,
  cancelHatch,
  cleanPetName,
  cleanupPetGen,
  closePetGenerate,
  discardHatched,
  generateDrafts,
  hatchSelected
} from '@/store/pet-generate'

const VARIANT_COUNT = 4
const PREVIEW_SCALE = 0.7
const PREVIEW_ROWS = [
  'idle',
  'waving',
  'running-right',
  'running-left',
  'running',
  'review',
  'jumping',
  'failed',
  'waiting'
]
const PREVIEW_STATE_MS = 1400

const ROW_TO_FRAME_KEY: Record<string, string> = {
  idle: 'idle',
  wave: 'wave',
  waving: 'wave',
  jump: 'jump',
  jumping: 'jump',
  run: 'run',
  running: 'run',
  'running-right': 'run',
  'running-left': 'run',
  failed: 'failed',
  review: 'review',
  waiting: 'waiting'
}

function frameCountForRow(pet: PetInfo, row: string): number {
  const byState = pet.framesByState
  const mapped = ROW_TO_FRAME_KEY[row]
  return byState?.[row] ?? (mapped ? byState?.[mapped] : undefined) ?? pet.framesPerState ?? 0
}

export function PetGenerateOverlay() {
  const open = useStore($petGenerateOpen)
  const status = useStore($petGenStatus)
  const { requestGateway } = useGatewayRequest()

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      // Deletes a hatched-but-unadopted preview pet so it doesn't linger, then
      // resets all generation state.
      cleanupPetGen(requestGateway)
      closePetGenerate()
    }
  }

  // The draft screen needs room for the 2×2 grid; the single-pet screens
  // (hatch egg, reveal) shrink to the pet's frame so it isn't lost in a wide box.
  const single = status === 'hatching' || status === 'preview' || status === 'adopting'

  return (
    <Dialog onOpenChange={handleOpenChange} open={open}>
      <DialogContent
        aria-describedby={undefined}
        className={cn('max-w-none gap-4 text-center', single ? 'w-[min(17rem,92vw)]' : 'w-[min(23rem,92vw)]')}
      >
        {open && <PetGenerateContent />}
      </DialogContent>
    </Dialog>
  )
}

function PetGenerateContent() {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet
  const { requestGateway } = useGatewayRequest()

  const status = useStore($petGenStatus)
  const error = useStore($petGenError)
  const drafts = useStore($petGenDrafts)
  const selected = useStore($petGenSelected)
  const preview = useStore($petGenPreview)
  const stage = useStore($petGenStage)

  const [prompt, setPrompt] = useState('')

  const busy = status === 'generating' || status === 'hatching'
  const hasDrafts = drafts.length > 0
  const generating = status === 'generating'
  // The idle "describe a pet" state — egg + suggestions get generous, equidistant
  // breathing room (gap-7.5) from the prompt; the working states stay compact.
  const isEmptyState =
    !hasDrafts &&
    !generating &&
    status !== 'hatching' &&
    status !== 'preview' &&
    status !== 'adopting' &&
    status !== 'stale'

  const close = () => {
    cleanupPetGen(requestGateway)
    closePetGenerate()
  }

  const generate = () => {
    if (prompt.trim() && !busy) {
      void generateDrafts(requestGateway, { prompt: prompt.trim() })
    }
  }

  // One-click an example prompt straight into a draft round.
  const runExample = (example: string) => {
    setPrompt(example)
    void generateDrafts(requestGateway, { prompt: example })
  }

  // Hatch with a clean default name derived from the prompt (the prompt itself
  // is grounding text, not a label); the user names it on the reveal screen.
  const hatch = () => {
    if (prompt.trim()) {
      void hatchSelected(requestGateway, { name: cleanPetName(prompt), prompt: prompt.trim() })
    }
  }

  const adopt = (finalName: string) => {
    void adoptHatched(requestGateway, finalName).then(out => {
      if (out.ok) {
        triggerHaptic('crisp')
        close()
      }
    })
  }

  // The header title tracks the phase instead of sticking on "Generate a pet".
  const headerTitle =
    status === 'hatching' ? copy.spawning : status === 'preview' || status === 'adopting' ? copy.hatched : copy.title
  // Prompt input only belongs on the describe/draft screens.
  const showPrompt = status !== 'hatching' && status !== 'preview' && status !== 'adopting'

  return (
    <>
      <DialogHeader>
        <DialogTitle icon={Egg}>{headerTitle}</DialogTitle>
      </DialogHeader>

      <div className={cn('flex min-h-0 flex-1 flex-col', isEmptyState ? 'gap-4' : 'gap-2.5')}>
        {/* Concept prompt with the inline sparkle generate/stop affordance (the
            same primitive as the commit-message + project-idea fields). */}
        {showPrompt && (
          <div className="relative">
            <Input
              autoFocus
              className="pr-9"
              onChange={event => setPrompt(event.target.value)}
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
              disabled={!prompt.trim()}
              generating={generating}
              generatingLabel={t.common.cancel}
              label={copy.generate}
              onCancel={cancelGenerate}
              onGenerate={generate}
            />
          </div>
        )}

        {error && status !== 'preview' && status !== 'adopting' && (
          <Alert variant="destructive">
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {status === 'stale' ? (
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
          <EmptyHint onExample={runExample} />
        ) : (
          <DraftGrid
            busy={busy}
            drafts={drafts}
            generating={generating}
            hasDrafts={hasDrafts}
            onHatch={hatch}
            onSelect={index => $petGenSelected.set(index)}
            selected={selected}
          />
        )}
      </div>
    </>
  )
}

// Creative seed prompts — specifics make better pets (petdex's own advice).
// Doubling as guidance and a one-click way to see the flow.
const EXAMPLE_PROMPTS = ['a bubble-tea otter', 'a tiny sock elf', 'a pixel dragon', 'a grumpy office cat', 'a neon axolotl']

function EmptyHint({ onExample }: { onExample: (prompt: string) => void }) {
  return (
    <div className="flex flex-col items-center gap-2">
      <p className="text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">Need a spark?</p>
      <div className="flex flex-wrap items-center justify-center gap-1.5">
        {EXAMPLE_PROMPTS.map(example => (
          <Button className="rounded-full" key={example} onClick={() => onExample(example)} size="xs" variant="outline">
            {example}
          </Button>
        ))}
      </div>
    </div>
  )
}

function HatchingView({ stage }: { stage: { phase: string; state?: string; done?: number; total?: number } | null }) {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet

  const subtitle = stage
    ? stage.phase === 'row'
      ? copy.hatchRow(stage.state ?? '', stage.done ?? 0, stage.total ?? 0)
      : stage.phase === 'compose'
        ? copy.hatchComposing
        : copy.hatchSaving
    : copy.hatchingSub

  return <PetEggHatch cancelLabel={t.common.cancel} onCancel={cancelHatch} subtitle={subtitle} />
}

interface DraftGridProps {
  busy: boolean
  drafts: { index: number; dataUri: string }[]
  generating: boolean
  hasDrafts: boolean
  onHatch: () => void
  onSelect: (index: number) => void
  selected: number | null
}

function DraftGrid({ busy, drafts, generating, hasDrafts, onHatch, onSelect, selected }: DraftGridProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet

  const slots = generating
    ? Array.from({ length: VARIANT_COUNT }, (_, i) => drafts.find(draft => draft.index === i) ?? null)
    : drafts

  return (
    <div className="flex flex-col gap-2">
      {generating && (
        <div className="flex items-center justify-between text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
          <span className="shimmer">{copy.generating}</span>
          <span className="tabular-nums">
            {drafts.length}/{VARIANT_COUNT}
          </span>
        </div>
      )}

      <div className="grid grid-cols-2 gap-2">
        {slots.map((draft, i) => {
          const isSelected = !generating && draft != null && selected === draft.index

          return (
            <button
              className={cn(
                'relative flex aspect-[192/208] items-center justify-center overflow-hidden',
                selectableCardClass({ active: isSelected, prominent: true })
              )}
              disabled={generating || busy || draft == null}
              key={draft ? `draft-${draft.index}` : `slot-${i}`}
              onClick={() => draft != null && onSelect(draft.index)}
              type="button"
            >
              {draft != null ? (
                // Hatches into place as each draft streams back.
                <img alt="" className="pet-reveal size-full object-contain p-1.5" draggable={false} src={draft.dataUri} />
              ) : (
                // Incubating: a creme egg resting on its contact shadow.
                <div className="relative z-10 flex flex-col items-center">
                  <PixelEggSprite index={i} mode="bounce" size={48} />
                  <span className="pet-egg-shadow pet-egg-shadow--sm mt-1" />
                </div>
              )}
            </button>
          )
        })}
      </div>

      {hasDrafts && (
        <Button className="w-full" disabled={busy || selected === null} onClick={onHatch}>
          <PawPrint />
          {copy.hatch}
        </Button>
      )}
    </div>
  )
}

interface HatchPreviewProps {
  pet: PetInfo
  adopting: boolean
  error: string | null
  onAdopt: (name: string) => void
  onDiscard: () => void
}

function HatchPreview({ pet, adopting, error, onAdopt, onDiscard }: HatchPreviewProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generatePet
  // Empty so the "Name your pet" placeholder shows; blank adopt keeps the
  // provisional name from the prompt.
  const [name, setName] = useState('')
  // Play the egg's crack/hatch frames once before swapping in the live pet.
  const [revealed, setRevealed] = useState(false)
  // Right after the egg cracks the pet plays its "yay" jump a couple times, then
  // hands off to the normal state-cycling preview.
  const [celebrating, setCelebrating] = useState(false)
  const [stateIndex, setStateIndex] = useState(0)
  const previewRows = (pet.stateRows?.length ? pet.stateRows : PREVIEW_ROWS).filter(row => frameCountForRow(pet, row) > 0)
  const rows = previewRows.length > 0 ? previewRows : ['idle']
  const activeRow = rows[stateIndex % rows.length] ?? 'idle'
  const canJump = frameCountForRow(pet, 'jumping') > 0
  const rowOverride = celebrating && canJump ? 'jumping' : activeRow

  useEffect(() => {
    const id = setInterval(() => setStateIndex(i => (i + 1) % rows.length), PREVIEW_STATE_MS)
    return () => clearInterval(id)
  }, [rows.length])

  // On reveal: celebrate (jump) ~2 loops, then drop into the cycling preview.
  useEffect(() => {
    if (!revealed) {
      return
    }
    setCelebrating(true)
    const id = setTimeout(() => {
      setCelebrating(false)
      setStateIndex(0)
    }, 2 * (pet.loopMs ?? 1100))
    return () => clearTimeout(id)
  }, [revealed, pet.loopMs])

  useEffect(() => {
    setStateIndex(0)
    setName('')
    setRevealed(false)
    setCelebrating(false)
  }, [pet.slug])

  const previewInfo: PetInfo = { ...pet, scale: PREVIEW_SCALE }

  return (
    <div className="flex flex-col items-center gap-2 py-1">
      {/* Fills the (now narrow) dialog so the pet frame is the screen width. */}
      <div className="relative flex aspect-[192/208] w-full items-center justify-center overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary)">
        {revealed ? (
          <>
            <div className="pet-reveal">
              <PetSprite info={previewInfo} rowOverride={rowOverride} />
            </div>
            <PetStarShower />
          </>
        ) : (
          // The egg cracks open, then we swap in the live pet.
          <PixelEggSprite
            mode="hatch"
            onDone={() => {
              setRevealed(true)
              triggerHaptic('crisp')
            }}
            size={150}
          />
        )}
      </div>

      <Input
        autoFocus
        className="w-full"
        onChange={event => setName(event.target.value)}
        onKeyDown={event => {
          if (event.key === 'Enter') {
            event.preventDefault()
            onAdopt(name)
          }
        }}
        placeholder={copy.namePlaceholder}
        value={name}
      />

      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      <div className="flex w-full items-center gap-1.5">
        <Button disabled={adopting} onClick={onDiscard} variant="ghost">
          <RefreshCw />
          {copy.startOver}
        </Button>
        <Button className="flex-1" disabled={adopting} onClick={() => onAdopt(name)}>
          {adopting ? <Loader2 className="animate-spin" /> : <PawPrint />}
          {copy.adopt}
        </Button>
      </div>
    </div>
  )
}

