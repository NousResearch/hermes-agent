/**
 * "Hatch a Pet" — a dedicated, Pokédex-style overlay for pet generation.
 *
 * Previously generation lived as a cramped nested page inside the Cmd-K command
 * palette (~34rem popover). This is its own full Radix dialog with room to
 * breathe: a device-framed header, its own concept prompt, a roomy draft grid
 * that streams in live, and the egg-hatch + reveal flow. It's a thin view over
 * the `pet-generate` store; the store owns the generate → hatch → adopt steps.
 *
 * This file is just the dialog shell + sizing; the flow lives in
 * `PetGenerateContent`, and each screen is its own atomic component under
 * `./components`.
 */

import { useStore } from '@nanostores/react'

import { useGatewayRequest } from '@/app/gateway/hooks/use-gateway-request'
import { useRouteOverlayActive } from '@/app/hooks/use-route-overlay-active'
import { Dialog, DialogContent } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'
import {
  $petGenDrafts,
  $petGenerateOpen,
  $petGenError,
  $petGenStatus,
  cleanupPetGenOnClose,
  closePetGenerate
} from '@/store/pet-generate'

import { PetGenerateContent } from './pet-generate-content'

export function PetGenerateOverlay() {
  const { t } = useI18n()
  const { requestGateway } = useGatewayRequest()
  const open = useStore($petGenerateOpen)
  const status = useStore($petGenStatus)
  const error = useStore($petGenError)
  const drafts = useStore($petGenDrafts)

  // Yield the screen to a full-screen route overlay (e.g. /settings while the
  // user adds an image-gen key) without tearing down — the store keeps us open,
  // and we reappear + re-check on return.
  if (useRouteOverlayActive()) {
    return null
  }

  const handleOpenChange = (next: boolean) => {
    if (!next) {
      cleanupPetGenOnClose(requestGateway)
      // Never interrupt in-flight work. Generating/hatching continues in the
      // background; only an unadopted finished preview is discarded on close.
      closePetGenerate()
    }
  }

  // The draft screen needs room for the 2×2 grid; the single-pet screens
  // (hatch egg, reveal) shrink to the pet's frame so it isn't lost in a wide box.
  // `fitContent` lets the dialog size to content; the `min-w` floors each phase.
  const single = status === 'hatching' || status === 'preview' || status === 'adopting'
  const copy = t.commandCenter.generatePet

  // The footer banner narrates the dialog's async state: the failure reason on a
  // dead-end error, else the "you can close this, we'll notify you" reassurance
  // while a generate/hatch runs in the background. On step 1, show a neutral ETA.
  const working = status === 'generating' || status === 'hatching'
  const errored = status === 'error' && drafts.length === 0
  const stepOne = status === 'idle' || status === 'ready'

  const banner = errored
    ? error || copy.genericError
    : working
      ? copy.backgroundHint
      : stepOne
        ? copy.slowProviderHint
        : undefined

  return (
    <Dialog onOpenChange={handleOpenChange} open={open}>
      <DialogContent
        aria-describedby={undefined}
        banner={banner}
        bannerTone={errored ? 'error' : 'info'}
        // Cap the width so a long banner (e.g. a provider refusal) wraps instead
        // of stretching the dialog out; the min-w floors each phase.
        className={cn('gap-4 text-center', single ? 'min-w-[17rem] max-w-[20rem]' : 'min-w-[19rem] max-w-[22rem]')}
        fitContent
      >
        {open && <PetGenerateContent />}
      </DialogContent>
    </Dialog>
  )
}
