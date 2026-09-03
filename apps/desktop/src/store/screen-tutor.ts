import { atom } from 'nanostores'

import type { ComposerDraft } from '@/app/chat/composer/contrib'
import type { ScreenTutorOverlayState } from '@/global'
import { createComposerAttachmentOccurrenceId } from '@/store/composer'

export interface ScreenTutorState {
  armedTarget: string | null
  error: string | null
  overlay: ScreenTutorOverlayState
  status: 'idle' | 'capturing'
}

const EMPTY_OVERLAY = { count: 0, frozen: false, visible: false }

export const $screenTutor = atom<ScreenTutorState>({
  armedTarget: null,
  error: null,
  overlay: EMPTY_OVERLAY,
  status: 'idle'
})

export function toggleScreenTutor(target: string): void {
  const current = $screenTutor.get()
  $screenTutor.set({
    armedTarget: current.armedTarget === target ? null : target,
    error: null,
    overlay: current.overlay,
    status: 'idle'
  })
}

export function armScreenTutor(target: string): void {
  const current = $screenTutor.get()
  $screenTutor.set({ ...current, armedTarget: target, error: null, status: 'idle' })
}

export async function prepareScreenTutorDraft(target: string, draft: ComposerDraft): Promise<ComposerDraft | null> {
  const bridge = window.hermesDesktop?.screenTutor
  const current = $screenTutor.get()

  if (!bridge || current.armedTarget !== target) {
    return draft
  }

  $screenTutor.set({ armedTarget: null, error: null, overlay: current.overlay, status: 'capturing' })

  try {
    const capture = await bridge.capture()
    const visibleText = draft.displayText ?? draft.text

    const instructions = [
      '[Screen Tutor mode]',
      `The user's visible request is: ${JSON.stringify(visibleText)}`,
      `A fresh screenshot of display ${capture.display.id} (${capture.display.label}) is attached.`,
      'Analyze the screenshot and answer the request in clear, short steps.',
      'When visual guidance materially improves the answer, call screen_tutor_annotate once with 1–8 concise annotations using normalized coordinates (0 at top-left, 1 at bottom-right). Use arrows/lines for direction, rect/circle for regions, point for one control, and labels sparingly.',
      'If the user asks to learn or do a task step by step, show exactly one next action and include guide metadata. Reuse one guide id, state the visible success check, and keep the overlay frozen while waiting for the user.',
      'For Check step, compare the fresh screenshot with the current success check. Advance only when the expected result is visibly present; otherwise keep the same step and annotate the correction.',
      'Use screen_tutor_point only when exactly one pointer is enough. Never call both tools for the same answer.',
      'Do not guess coordinates. If a target is not visible or uncertain, explain that instead and do not annotate it.',
      'Annotations are read-only visual guidance. They cannot click, type, trade, or operate the other app.'
    ].join('\n')

    return {
      attachments: [
        ...(draft.attachments ?? []),
        {
          detail: `${capture.image.width}×${capture.image.height} · ${capture.display.label}`,
          id: `screen-tutor:${capture.path}`,
          kind: 'image',
          label: 'Screen Tutor capture',
          occurrenceId: createComposerAttachmentOccurrenceId(),
          path: capture.path
        }
      ],
      displayText: visibleText,
      text: `${visibleText}\n\n${instructions}`
    }
  } catch (error) {
    $screenTutor.set({
      armedTarget: target,
      error: error instanceof Error ? error.message : String(error),
      overlay: $screenTutor.get().overlay,
      status: 'idle'
    })

    return null
  } finally {
    if ($screenTutor.get().status === 'capturing') {
      $screenTutor.set({ armedTarget: null, error: null, overlay: $screenTutor.get().overlay, status: 'idle' })
    }
  }
}

export function resetScreenTutor(): void {
  $screenTutor.set({ armedTarget: null, error: null, overlay: $screenTutor.get().overlay, status: 'idle' })
}

export function dismissScreenAnnotations(): void {
  window.hermesDesktop?.screenTutor?.dismiss()
  $screenTutor.set({ ...$screenTutor.get(), overlay: EMPTY_OVERLAY })
}

export function toggleScreenAnnotationsFrozen(): void {
  const current = $screenTutor.get()

  if (!current.overlay.visible) {
    return
  }

  const frozen = !current.overlay.frozen
  window.hermesDesktop?.screenTutor?.setFrozen(frozen)
  $screenTutor.set({ ...current, overlay: { ...current.overlay, frozen } })
}

window.hermesDesktop?.screenTutor?.onState(overlay => {
  $screenTutor.set({ ...$screenTutor.get(), overlay })
})
