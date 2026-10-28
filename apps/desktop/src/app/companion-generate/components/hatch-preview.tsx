import { useEffect, useState } from 'react'

import { CompanionSprite } from '@/components/companion/companion-sprite'
import { CompanionStarShower } from '@/components/companion/companion-star-shower'
import { PixelEggSprite } from '@/components/companion/pixel-egg-sprite'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Loader2, PawPrint, RefreshCw } from '@/lib/icons'
import { isSubmitEnter } from '@/lib/ime'
import { type CompanionInfo } from '@/store/companion'

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
  companion: CompanionInfo
  adopting: boolean
  error: string | null
  onAdopt: (name: string) => void
  onDiscard: () => void
}

export function HatchPreview({ companion, adopting, error, onAdopt, onDiscard }: HatchPreviewProps) {
  const { t } = useI18n()
  const copy = t.commandCenter.generateCompanion
  // Empty so the "Name your companion" placeholder shows; blank adopt keeps the
  // provisional name from the prompt.
  const [name, setName] = useState('')
  // Play the egg's crack/hatch frames once before swapping in the live companion.
  const [revealed, setRevealed] = useState(false)
  // Right after the egg cracks the companion plays its "yay" jump a couple times, then
  // hands off to the normal state-cycling preview.
  const [celebrating, setCelebrating] = useState(false)
  const [stateIndex, setStateIndex] = useState(0)

  const previewRows = (companion.stateRows?.length ? companion.stateRows : PREVIEW_ROWS).filter(
    row => frameCountForRow(companion, row) > 0
  )

  const rows = previewRows.length > 0 ? previewRows : ['idle']
  const activeRow = rows[stateIndex % rows.length] ?? 'idle'
  const canJump = frameCountForRow(companion, 'jumping') > 0
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
      2 * (companion.loopMs ?? 1100)
    )

    return () => clearTimeout(id)
  }, [revealed, companion.loopMs])

  useEffect(() => {
    setStateIndex(0)
    setName('')
    setRevealed(false)
    setCelebrating(false)
  }, [companion.slug])

  const previewInfo: CompanionInfo = { ...companion, scale: PREVIEW_SCALE }

  return (
    <div className="flex flex-col items-center gap-2">
      {/* Fills the (now narrow) dialog so the companion frame is the screen width. */}
      <div className="relative flex aspect-[192/208] w-full items-center justify-center overflow-hidden rounded-lg border border-(--ui-stroke-tertiary) bg-(--ui-bg-quinary)">
        {revealed ? (
          <>
            <div className="relative inline-block">
              <span aria-hidden className="companion-contact-shadow" />
              <div className="companion-reveal relative z-10">
                <CompanionSprite info={previewInfo} rowOverride={rowOverride} />
              </div>
            </div>
            <CompanionStarShower />
          </>
        ) : (
          // The egg cracks open, then we swap in the live companion.
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
          if (isSubmitEnter(event)) {
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
