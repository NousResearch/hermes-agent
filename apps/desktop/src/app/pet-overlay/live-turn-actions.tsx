import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import type { Translations } from '@/i18n'
import type { PetActionCenterItem } from '@/store/pet-action-center'

type ActionCenterStrings = Translations['pet']['actionCenter']
type LiveTurnItem = Extract<PetActionCenterItem, { kind: 'live-turn' }>
type LiveTextAction = 'queue' | 'send' | 'steer'

export interface LiveTurnActionsProps {
  ac: ActionCenterStrings
  draft: string
  isSteerRejected: boolean
  isSubmitting: boolean
  item: LiveTurnItem
  onAcknowledge: () => void
  onDraftChange: (value: string) => void
  onStop: () => void
  onTextAction: (action: LiveTextAction) => void
}

function hasAction(item: LiveTurnItem, action: LiveTurnItem['allowedActions'][number]): boolean {
  return item.allowedActions.includes(action)
}

function defaultTextAction(item: LiveTurnItem, isSteerRejected: boolean): LiveTextAction | null {
  if (isSteerRejected) {
    return null
  }

  if (item.status === 'idle' && hasAction(item, 'send')) {
    return 'send'
  }

  if (item.status === 'working' || item.status === 'reviewing') {
    if (hasAction(item, 'steer')) {
      return 'steer'
    }

    if (hasAction(item, 'queue')) {
      return 'queue'
    }
  }

  return null
}

export function LiveTurnActions({
  ac,
  draft,
  isSteerRejected,
  isSubmitting,
  item,
  onAcknowledge,
  onDraftChange,
  onStop,
  onTextAction
}: LiveTurnActionsProps) {
  const idle = item.status === 'idle'
  const active = item.status === 'working' || item.status === 'reviewing'
  const canSend = idle && hasAction(item, 'send')
  const canSteer = active && hasAction(item, 'steer')
  const canQueue = active && hasAction(item, 'queue')
  const canStop = active && hasAction(item, 'stop')
  const canAcknowledge = (item.status === 'done' || item.status === 'failed') && hasAction(item, 'acknowledge')
  const showDraft = canSend || canSteer || canQueue
  const trimmedDraft = draft.trim()
  const enterAction = defaultTextAction(item, isSteerRejected)

  function onKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (event.nativeEvent.isComposing || event.key !== 'Enter' || event.shiftKey) {
      return
    }

    if (enterAction && trimmedDraft && !isSubmitting) {
      event.preventDefault()
      onTextAction(enterAction)
    }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      {showDraft && (
        <Textarea
          aria-label={canQueue && !canSteer && !canSend ? ac.queueMessage : ac.messageSession}
          disabled={isSubmitting}
          onChange={event => onDraftChange(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder={ac.liveMessagePlaceholder}
          rows={2}
          size="sm"
          value={draft}
        />
      )}

      {(canSend || canSteer || canQueue || canStop || canAcknowledge) && (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4, justifyContent: 'flex-end' }}>
          {canSend && (
            <Button
              disabled={!trimmedDraft || isSubmitting}
              onClick={() => onTextAction('send')}
              size="xs"
              variant="default"
            >
              {ac.send}
            </Button>
          )}
          {canSteer && (
            <Button
              disabled={!trimmedDraft || isSubmitting}
              onClick={() => onTextAction('steer')}
              size="xs"
              variant="default"
            >
              {ac.steer}
            </Button>
          )}
          {canQueue && (
            <Button
              disabled={!trimmedDraft || isSubmitting}
              onClick={() => onTextAction('queue')}
              size="xs"
              variant="secondary"
            >
              {isSteerRejected ? ac.queueRejected : ac.queue}
            </Button>
          )}
          {canStop && (
            <Button disabled={isSubmitting} onClick={onStop} size="xs" variant="destructive">
              {ac.stop}
            </Button>
          )}
          {canAcknowledge && (
            <Button disabled={isSubmitting} onClick={onAcknowledge} size="xs" variant="secondary">
              {ac.acknowledge}
            </Button>
          )}
        </div>
      )}
    </div>
  )
}

export type { LiveTextAction, LiveTurnItem }
