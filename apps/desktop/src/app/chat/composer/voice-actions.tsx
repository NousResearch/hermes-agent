import type { ReactNode } from 'react'

import { Slot as ContribSlot } from '@/contrib/react/slot'

import { COMPOSER_AREAS } from './contrib'
import { type ComposerVoiceController, ComposerVoiceControllerProvider } from './hooks/use-composer-voice'

export function ComposerVoiceActions({
  children,
  controller
}: {
  children: ReactNode
  controller: ComposerVoiceController
}) {
  return (
    <div className="flex min-w-0 items-center justify-end gap-(--composer-control-gap) [grid-area:controls]">
      <ComposerVoiceControllerProvider controller={controller}>
        <ContribSlot area={COMPOSER_AREAS.actions} />
      </ComposerVoiceControllerProvider>
      {children}
    </div>
  )
}
