import type { BusyInputMode } from '@/store/busy-input-mode'

import { busyModeHasCarrier, resolveBusyComposerAction, resolveComposerHintActions } from './busy-input-mode'

export type ComposerBusyAction = 'interrupt' | 'steer' | 'queue' | 'stop'

interface BusyComposerInputs {
  attachmentCount: number
  blockingPrompt: boolean
  busy: boolean
  busyInputMode: BusyInputMode | null
  cardWaiting: boolean
  compacting: boolean
  hasComposerPayload: boolean
  hasHiddenSteerHandler: boolean
  hasSteerHandler: boolean
  hasText: boolean
  isSteerableText: boolean
  queueEditing: boolean
  replyPaused: boolean
}

/** Resolve controls and help together, using the carrier each action will call. */
export function deriveBusyComposerState(inputs: BusyComposerInputs) {
  const { attachmentCount, blockingPrompt, busy, busyInputMode, cardWaiting, compacting, isSteerableText } = inputs

  const canCorrectPayload =
    busy && !cardWaiting && !blockingPrompt && !inputs.replyPaused && !compacting && attachmentCount === 0 && isSteerableText

  const canSubmit = (busy || inputs.hasComposerPayload) && !(canCorrectPayload && busyInputMode === null)
  const canSteer = canCorrectPayload && inputs.hasSteerHandler

  const canCorrect =
    canCorrectPayload &&
    busyInputMode !== null &&
    busyModeHasCarrier(busyInputMode, {
      onSteer: inputs.hasSteerHandler,
      onSteerHidden: inputs.hasHiddenSteerHandler
    })

  const action = cardWaiting
    ? 'steer'
    : resolveBusyComposerAction({
        busy,
        canCorrect,
        compacting,
        hasPayload: inputs.hasComposerPayload,
        blockingPrompt,
        mode: busyInputMode ?? 'interrupt'
      })

  const hints = resolveComposerHintActions({
    canSubmit,
    directSubmit: inputs.queueEditing || cardWaiting || (inputs.hasText && !isSteerableText && attachmentCount === 0),
    action,
    mode: busyInputMode,
    alternateCanRedirect: canSteer
  })

  return { busyAction: action === 'redirect' ? ('interrupt' as const) : action, canSteer, canSubmit, ...hints }
}
