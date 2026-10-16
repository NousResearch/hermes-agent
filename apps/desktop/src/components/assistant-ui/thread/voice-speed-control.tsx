import { useStore } from '@nanostores/react'
import { type FC } from 'react'

import { TooltipIconButton } from '@/components/assistant-ui/tooltip-icon-button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { GaugeIcon } from '@/lib/icons'
import { cn } from '@/lib/utils'
import {
  $voicePlaybackSpeed,
  DEFAULT_VOICE_PLAYBACK_SPEED,
  setVoicePlaybackSpeed,
  VOICE_PLAYBACK_SPEED_PRESETS
} from '@/store/voice-playback-speed'

const formatSpeed = (speed: number): string => `${speed}×`

// Speech playback speed for this device, Slack-style: one control that governs
// every read-aloud and voice-conversation reply, not per-message state. Lives
// next to the read-aloud button it modulates and shows the active rate once
// it departs from 1×. The preference persists ($voicePlaybackSpeed) and every
// later playback — including one already speaking — picks it up.
export const VoiceSpeedControl: FC = () => {
  const { t } = useI18n()
  const copy = t.assistant.thread
  const speed = useStore($voicePlaybackSpeed)

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <TooltipIconButton
          data-testid="voice-speed-control"
          onClick={() => triggerHaptic('selection')}
          tooltip={copy.playbackSpeed}
        >
          <span className="flex items-center gap-0.5">
            <GaugeIcon className={cn('size-3.5', speed !== DEFAULT_VOICE_PLAYBACK_SPEED && 'text-foreground')} />
            {speed !== DEFAULT_VOICE_PLAYBACK_SPEED && (
              <span className="text-[0.625rem] tabular-nums leading-none">{formatSpeed(speed)}</span>
            )}
          </span>
        </TooltipIconButton>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="min-w-0">
        {VOICE_PLAYBACK_SPEED_PRESETS.map(preset => (
          <DropdownMenuItem
            className="justify-center tabular-nums"
            data-testid={`voice-speed-${preset}`}
            key={preset}
            onSelect={() => {
              triggerHaptic('selection')
              setVoicePlaybackSpeed(preset)
            }}
          >
            {formatSpeed(preset)}
            {preset === DEFAULT_VOICE_PLAYBACK_SPEED && (
              <span className="text-muted-foreground">{` — ${copy.playbackSpeedNormal}`}</span>
            )}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )
}
