import { Button } from '@/components/ui/button'
import { DropdownMenu, DropdownMenuContent, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { Tip } from '@/components/ui/tooltip'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { AudioLines, ChevronDown, iconSize } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { GHOST_ICON_BTN, PRIMARY_ICON_BTN } from './control-classes'
import { useVoiceEngineName, VoiceEngineRows } from './voice-engine-rows'

/**
 * The primary "start voice conversation" button, with the engine picker one
 * click away when the layout shows the voice controls unfolded.
 *
 * In the folded layout the picker lives in the voice menu; unfolded there is
 * no menu, so without this the only way to swap engines was Settings → Voice,
 * which is not where you are when you want to talk. The chevron is a separate
 * button so the primary press stays a single unambiguous action; the tooltip
 * names the engine so the choice is visible before pressing.
 */
export function StartVoiceButton({ disabled, label, onStart }: { disabled: boolean; label: string; onStart: () => void }) {
  const { t } = useI18n()
  const engine = useVoiceEngineName()

  return (
    <span className="flex items-center">
      <Tip label={engine ? `${label} — ${engine}` : label}>
        <Button
          aria-label={label}
          className={cn(PRIMARY_ICON_BTN, engine && 'rounded-r-none')}
          disabled={disabled}
          onClick={() => {
            triggerHaptic('open')
            onStart()
          }}
          size="icon"
          type="button"
        >
          <AudioLines className={iconSize.sm} />
        </Button>
      </Tip>
      {engine ? (
        <DropdownMenu>
          <Tip label={t.composer.voiceEngine}>
            <DropdownMenuTrigger asChild>
              <Button
                aria-label={t.composer.voiceEngine}
                className={cn(GHOST_ICON_BTN, 'w-5 rounded-l-none p-0')}
                disabled={disabled}
                size="icon"
                type="button"
                variant="ghost"
              >
                <ChevronDown className={iconSize.xs} />
              </Button>
            </DropdownMenuTrigger>
          </Tip>
          <DropdownMenuContent align="end" className="min-w-52">
            <VoiceEngineRows disabled={disabled} />
          </DropdownMenuContent>
        </DropdownMenu>
      ) : null}
    </span>
  )
}
