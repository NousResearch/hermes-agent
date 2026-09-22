import type { BotRequirementReadiness } from '@hermes/shared'

import { Button } from '@/components/ui/button'

import type { InstalledBot } from './bot-types'

export interface ReadinessCopy {
  ready: string
  needsSetup: string
  runtime: string
  firstTask: string
  firstTaskComplete: string
  firstTaskPending: string
  finishSetup: string
  checkAgain: string
  rechecking: string
  startFirstTask: string
  open: string
  reusedSignIn: string
  setupAction: (kind: string, id: string) => string
  samples: string
  samplesHint: string
}

interface ReadinessPanelProps {
  bot: InstalledBot
  busy: boolean
  copy: ReadinessCopy
  onOpen: () => void
  onRefresh: () => void
  onSetupAction: (requirement: BotRequirementReadiness | { action: string; id: string }) => void
  onStartFirstTask: () => void
}

function StateMark({ ready }: { ready: boolean }) {
  return (
    <span aria-hidden className={ready ? 'text-(--ui-success)' : 'text-(--ui-warning)'}>
      {ready ? '✓' : '○'}
    </span>
  )
}

export function ReadinessPanel({
  bot,
  busy,
  copy,
  onOpen,
  onRefresh,
  onSetupAction,
  onStartFirstTask
}: ReadinessPanelProps) {
  const { status } = bot
  const runtimeReady = status.runtime.ok
  const firstTaskReady = status.first_task.status === 'complete'

  return (
    <div aria-label={`${bot.profile.name} ${copy.finishSetup}`} className="space-y-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="text-sm font-semibold text-(--ui-text-primary)">{bot.entry.title}</div>
          <div className="mt-1 text-xs text-(--ui-text-secondary)">{bot.entry.summary}</div>
        </div>
        <div className="text-xs text-(--ui-text-tertiary)">
          {status.setup_state === 'ready' ? copy.ready : copy.needsSetup}
        </div>
      </div>

      <div className="space-y-2 text-xs">
        <div className="flex items-start gap-2">
          <StateMark ready={runtimeReady} />
          <div className="min-w-0 flex-1">
            <div className="font-medium text-(--ui-text-primary)">{copy.runtime}</div>
            <div className="text-(--ui-text-secondary)">
              {runtimeReady
                ? [status.runtime.provider, status.runtime.model].filter(Boolean).join(' · ')
                : status.runtime.error}
            </div>
            {status.runtime.reused_sign_in && <div className="text-(--ui-text-tertiary)">{copy.reusedSignIn}</div>}
          </div>
          {!runtimeReady && status.runtime.action && (
            <Button
              aria-label={copy.setupAction(status.runtime.action, status.runtime.model ?? '')}
              onClick={() => onSetupAction({ action: status.runtime.action!, id: status.runtime.model ?? '' })}
              size="xs"
              variant="secondary"
            >
              {copy.setupAction(status.runtime.action, status.runtime.model ?? '')}
            </Button>
          )}
        </div>

        {(status.requirements ?? []).map(requirement => {
          const ready = requirement.status === 'ready'

          return (
            <div className="flex items-start gap-2" key={`${requirement.kind}:${requirement.id}`}>
              <StateMark ready={ready} />
              <div className="min-w-0 flex-1">
                <div className="font-medium text-(--ui-text-primary)">{requirement.id}</div>
                <div className="text-(--ui-text-secondary)">{requirement.detail ?? requirement.purpose}</div>
              </div>
              {!ready && requirement.action && (
                <Button
                  aria-label={copy.setupAction(requirement.kind, requirement.id)}
                  onClick={() => onSetupAction(requirement)}
                  size="xs"
                  variant="secondary"
                >
                  {copy.setupAction(requirement.kind, requirement.id)}
                </Button>
              )}
            </div>
          )
        })}

        <div className="flex items-start gap-2">
          <StateMark ready={firstTaskReady} />
          <div className="min-w-0 flex-1">
            <div className="font-medium text-(--ui-text-primary)">{copy.firstTask}</div>
            <div className="text-(--ui-text-secondary)">
              {firstTaskReady ? copy.firstTaskComplete : copy.firstTaskPending}
            </div>
          </div>
        </div>
      </div>

      {!firstTaskReady && status.starter_prompt && (
        <div className="space-y-1 border-l border-(--ui-stroke-tertiary) pl-3 text-xs">
          <div className="font-medium text-(--ui-text-secondary)">{copy.samples}</div>
          <div className="text-(--ui-text-primary)">{status.starter_prompt}</div>
          <div className="text-(--ui-text-tertiary)">{copy.samplesHint}</div>
        </div>
      )}

      <div className="flex flex-wrap gap-2">
        {status.can_start_first_task && !firstTaskReady && (
          <Button aria-label={`${copy.startFirstTask} ${bot.profile.name}`} onClick={onStartFirstTask} size="sm">
            {copy.startFirstTask}
          </Button>
        )}
        {firstTaskReady && (
          <Button aria-label={`${copy.open} ${bot.profile.name}`} onClick={onOpen} size="sm">
            {copy.open}
          </Button>
        )}
        <Button aria-label={`${busy ? copy.rechecking : copy.checkAgain} ${bot.profile.name}`} disabled={busy} onClick={onRefresh} size="sm" variant="secondary">
          {busy ? copy.rechecking : copy.checkAgain}
        </Button>
      </div>
    </div>
  )
}
