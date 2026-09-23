import { useStore } from '@nanostores/react'

import { SegmentedControl } from '@/components/ui/segmented-control'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import {
  $chatBubbles,
  CHAT_BUBBLE_FILL_MAX,
  CHAT_BUBBLE_FILL_MIN,
  CHAT_BUBBLE_FILL_STEP,
  type ChatBubbleCorners,
  type ChatBubbleTone,
  setChatBubbles
} from '@/store/chat-bubbles'
import { $userBubbleTransparency, setUserBubbleTransparency } from '@/store/user-bubble-transparency'

import { ControlRow, ListRow } from './primitives'
import { APPEARANCE_SETTING_IDS, appearanceSettingElementId } from './settings-search'

const CORNER_OPTIONS: ChatBubbleCorners[] = ['soft', 'round', 'pill']
const TONE_OPTIONS: ChatBubbleTone[] = ['theme', 'accent', 'neutral']

/**
 * Chat Layout — the transcript's shape, and everything that shapes it.
 *
 * One row (document vs bubbles) with the tuning panel underneath while bubbles
 * are on, so the idle settings page stays as short as it was: nothing here is
 * visible until the layout it tunes is chosen.
 *
 * The fill lever is deliberately the SAME state as the standalone
 * "Message Bubble" row above — `$userBubbleTransparency`. In the document
 * layout that lever means "your one bubble"; with bubbles on it means "the
 * bubbles", both sides, so it belongs beside the shapes it fills. Two stores
 * for one visual property is how they drift apart; one store reached from the
 * place the hand already is, is a shortcut, not a second source of truth.
 */
export function ChatBubbleSetting() {
  const { t } = useI18n()
  const a = t.settings.appearance
  const state = useStore($chatBubbles)
  const fill = useStore($userBubbleTransparency)
  const bubbles = state.layout === 'bubbles'

  const pick = (patch: Parameters<typeof setChatBubbles>[0]) => {
    triggerHaptic('selection')
    setChatBubbles(patch)
  }

  return (
    <ListRow
      action={
        <SegmentedControl
          onChange={id => pick({ layout: id })}
          options={[
            { id: 'document' as const, label: a.chatBubblesLayoutDocument },
            { id: 'bubbles' as const, label: a.chatBubblesLayoutBubbles }
          ]}
          value={state.layout}
        />
      }
      below={
        bubbles ? (
          <div className="mt-3 flex flex-col gap-2.5">
            <ControlRow label={a.chatBubblesCornersTitle}>
              <SegmentedControl
                onChange={id => pick({ corners: id })}
                options={CORNER_OPTIONS.map(id => ({ id, label: a.chatBubblesCorners[id] }))}
                value={state.corners}
              />
            </ControlRow>
            <ControlRow label={a.chatBubblesColorTitle}>
              <SegmentedControl
                onChange={id => pick({ tone: id })}
                options={TONE_OPTIONS.map(id => ({ id, label: a.chatBubblesColor[id] }))}
                value={state.tone}
              />
            </ControlRow>
            <ControlRow label={a.chatBubblesFillTitle}>
              <input
                aria-label={a.chatBubblesFillTitle}
                className="h-1 w-40 cursor-pointer appearance-none rounded-full bg-(--ui-stroke-tertiary)"
                max={CHAT_BUBBLE_FILL_MAX}
                min={CHAT_BUBBLE_FILL_MIN}
                onChange={event => {
                  triggerHaptic('selection')
                  setUserBubbleTransparency(Number(event.target.value))
                }}
                step={CHAT_BUBBLE_FILL_STEP}
                style={{ accentColor: 'var(--dt-primary)' }}
                type="range"
                value={fill}
              />
              <span className="w-9 text-right text-[length:var(--conversation-caption-font-size)] tabular-nums text-(--ui-text-tertiary)">
                {fill}%
              </span>
            </ControlRow>
          </div>
        ) : undefined
      }
      description={a.chatBubblesDesc}
      id={appearanceSettingElementId(APPEARANCE_SETTING_IDS.chatBubbles)}
      title={a.chatBubblesTitle}
    />
  )
}
