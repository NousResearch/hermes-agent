import { useStore } from '@nanostores/react'
import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { cn } from '@/lib/utils'
import { $screenTutor, armScreenTutor, dismissScreenAnnotations } from '@/store/screen-tutor'

import type { ChatBarProps } from './types'

type GuideAction = 'back' | 'check' | 'stuck'

const actionPrompt = {
  back: 'Go back one step. Inspect the fresh screenshot, restore the previous instruction, and annotate only that action.',
  check:
    'I completed this step. Inspect the fresh screenshot and verify the visible success check. Advance only if it is visibly satisfied; otherwise keep this step and show the correction.',
  stuck:
    'I am stuck on this step. Inspect the fresh screenshot, explain the blocker briefly, and redraw one clearer next action without taking control.'
} satisfies Record<GuideAction, string>

export function ScreenGuideCard({
  busy,
  disabled,
  onSubmit,
  target
}: {
  busy: boolean
  disabled: boolean
  onSubmit: ChatBarProps['onSubmit']
  target: string
}) {
  const state = useStore($screenTutor)
  const [sending, setSending] = useState<GuideAction | null>(null)
  const guide = state.overlay.guide

  if (!state.overlay.visible || !guide) {
    return null
  }

  const run = async (action: GuideAction) => {
    if (busy || disabled || sending) {
      return
    }

    setSending(action)
    armScreenTutor(target)

    const prompt = [
      '[Do It With Me guide control]',
      `Guide: ${guide.id}`,
      `Current step: ${guide.step} of ${guide.total}`,
      `Instruction: ${guide.instruction}`,
      guide.successCheck ? `Visible success check: ${guide.successCheck}` : '',
      actionPrompt[action]
    ]
      .filter(Boolean)
      .join('\n')

    try {
      await onSubmit(prompt, {
        displayText:
          action === 'check'
            ? `Check step ${guide.step}`
            : action === 'stuck'
              ? `I’m stuck on step ${guide.step}`
              : `Go back from step ${guide.step}`
      })
    } finally {
      setSending(null)
    }
  }

  const progress = `${Math.round((guide.step / guide.total) * 100)}%`

  return (
    <section
      aria-label={`${guide.title}, step ${guide.step} of ${guide.total}`}
      className="overflow-hidden rounded-lg border border-cyan-300/25 bg-[#111b27]/94 shadow-[0_12px_40px_rgba(0,0,0,0.32)] backdrop-blur-xl"
      data-screen-guide
    >
      <div className="h-0.5 bg-slate-700/70">
        <div
          className="h-full bg-cyan-300 transition-[width] duration-300 motion-reduce:transition-none"
          style={{ width: progress }}
        />
      </div>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2 px-3 py-2">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-cyan-200/35 bg-cyan-300/12 font-mono text-[11px] font-bold text-cyan-100">
          {guide.step}/{guide.total}
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate text-[10px] font-semibold tracking-[0.14em] text-cyan-200/70 uppercase">
            {guide.title}
          </div>
          <div className="truncate text-[12px] font-medium text-slate-100">{guide.instruction}</div>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-1">
          <Button
            aria-label="Go back one step"
            className="h-7 px-2 text-[11px]"
            disabled={busy || disabled || Boolean(sending) || guide.step <= 1}
            onClick={() => void run('back')}
            type="button"
            variant="ghost"
          >
            <Codicon name="arrow-left" />
          </Button>
          <Button
            className="h-7 px-2 text-[11px]"
            disabled={busy || disabled || Boolean(sending)}
            onClick={() => void run('stuck')}
            type="button"
            variant="ghost"
          >
            I’m stuck
          </Button>
          <Button
            className={cn('h-7 gap-1.5 px-2.5 text-[11px]', sending === 'check' && 'animate-pulse')}
            disabled={busy || disabled || Boolean(sending)}
            onClick={() => void run('check')}
            type="button"
          >
            <Codicon name="check" /> Check my step
          </Button>
          <Button
            aria-label="Stop guide"
            className="h-7 px-2 text-[11px]"
            onClick={dismissScreenAnnotations}
            type="button"
            variant="ghost"
          >
            <Codicon name="close" />
          </Button>
        </div>
      </div>
    </section>
  )
}
