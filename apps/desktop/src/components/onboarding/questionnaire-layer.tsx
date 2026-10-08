import { useStore } from '@nanostores/react'
import { lazy, Suspense } from 'react'

import { $questionnaireOpen } from '@/onboarding/store'

import { OverlaySurface } from './overlay-surface'

// Loaded on first use: the questionnaire pulls the layout and theme stores, and the overlay's
// picker and key form are also used by Settings, which needs none of that.
const QuestionnaireScreen = lazy(() =>
  import('@/onboarding/Questionnaire').then(module => ({ default: module.QuestionnaireScreen }))
)

/** The questionnaire as the first-run overlay's first state (D22). */
export function QuestionnaireLayer({
  refreshReadiness,
  statusbarVisible
}: {
  refreshReadiness: () => Promise<void>
  statusbarVisible: boolean
}) {
  const open = useStore($questionnaireOpen)

  if (!open) {
    return null
  }

  return (
    <OverlaySurface statusbarVisible={statusbarVisible}>
      <div className="relative flex max-h-full w-full max-w-[45rem] flex-col overflow-hidden rounded-xl border border-(--stroke-nous) bg-(--ui-chat-bubble-background) p-5 shadow-nous">
        <Suspense fallback={null}>
          <QuestionnaireScreen refreshReadiness={refreshReadiness} />
        </Suspense>
      </div>
    </OverlaySurface>
  )
}
