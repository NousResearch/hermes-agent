import { useStore } from '@nanostores/react'

import { Button } from '@/components/ui/button'
import { FadeScroll } from '@/components/ui/fade-scroll'
import { useI18n } from '@/i18n'
import { Loader2 } from '@/lib/icons'

import { chosenName } from '../flow'
import { handoffPrompt } from '../handoff'
import { $questionnaire } from '../store'

// Fits a full prompt (ask, About me, Before you start); a longer one scrolls with a fading edge as the cue.
const PROMPT_MAX_HEIGHT = '22rem'

/** The last screen: the first message as Hermes will get it, and Start. */
export function ReviewStep({ onStart }: { onStart: () => void }) {
  const { t } = useI18n()
  const { answers, facts, pending } = useStore($questionnaire)
  const copy = t.questionnaire.start
  const prompt = handoffPrompt(facts, answers)

  return (
    <div className="grid gap-4">
      <div className="grid gap-1">
        <h3 className="text-lg font-semibold tracking-tight">{copy.title(chosenName(answers))}</h3>
        <p className="text-sm text-muted-foreground">{copy.sub}</p>
      </div>
      <div className="rounded-lg bg-muted">
        <FadeScroll className="px-4 py-3" maxHeight={PROMPT_MAX_HEIGHT}>
          <pre className="font-sans text-[0.8125rem] leading-5 whitespace-pre-wrap text-foreground">{prompt}</pre>
        </FadeScroll>
      </div>
      <div className="flex justify-end">
        <Button disabled={pending !== null} onClick={onStart} type="button">
          {pending === 'start' ? (
            <>
              <Loader2 className="animate-spin" />
              {copy.waiting}
            </>
          ) : (
            copy.action
          )}
        </Button>
      </div>
    </div>
  )
}
