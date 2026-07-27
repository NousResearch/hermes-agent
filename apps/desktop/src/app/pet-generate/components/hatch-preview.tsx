import { useEffect, useState } from 'react'

import { PetSprite } from '@/components/pet/pet-sprite'
import { PetStarShower } from '@/components/pet/pet-star-shower'
import { PixelEggSprite } from '@/components/pet/pixel-egg-sprite'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2, PawPrint, RefreshCw } from '@/lib/icons'
import { type PetInfo } from '@/store/pet'

import { frameCountForRow } from '../lib/frame-count'

const PREVIEW_SCALE = 0.7
const PREVIEW_STATE_MS = 1400

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

interface HatchPreviewProps {
  pet: PetInfo
  adopting: boolean
  error: string | null
  onAdopt: (name: string) => void
  onDiscard: () => void
}

export function HatchPreview({ pet, adopting, error, onAdopt, onDiscard }: HatchPreviewProps) {
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

  const previewRows = (pet.stateRows?.length ? pet.stateRows : PREVIEW_ROWS).filter(
    row => frameCountForRow(pet, row) > 0
  )

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

    const id = setTimeout(
      () => {
        setCelebrating(false)
        setStateIndex(0)
      },
      2 * (pet.loopMs ?? 1100)
    )

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
    <div className="flex flex-col items-center gap-2">
      {/* Fills the (now narrow) dialog so the pet frame is the screen width. */}
      <div className="relative flex aspect-[192/208] w-full items-center justify-center overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary)">
        {revealed ? (
          <>
            <div className="relative inline-block">
              <span aria-hidden className="pet-contact-shadow" />
              <div className="pet-reveal relative z-10">
                <PetSprite info={previewInfo} rowOverride={rowOverride} />
              </div>
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
