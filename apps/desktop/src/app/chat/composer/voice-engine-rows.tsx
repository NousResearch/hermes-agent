import { useStore } from '@nanostores/react'

import {
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  dropdownMenuRow
} from '@/components/ui/dropdown-menu'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { notifyError } from '@/store/notifications'
import {
  $pluginVoiceEngineChoice,
  choosePluginVoiceEngine,
  resolvePluginVoiceEngine
} from '@/store/plugin-voice-engine'
import { $voiceLiveStatus, selectedVoiceChatMode, setVoiceChatMode } from '@/store/voice-live'

import { type RegisteredVoiceEngine, useComposerVoiceEngines } from './contrib'

const PLUGIN_PREFIX = 'plugin:'

/** The plugin engine the start button runs right now, or null for the core engine. */
export function useSelectedPluginVoiceEngine(): null | RegisteredVoiceEngine {
  const engines = useComposerVoiceEngines()
  const choice = useStore($pluginVoiceEngineChoice)

  return resolvePluginVoiceEngine(engines, choice)
}

/** The start-voice action: the chosen plugin engine's `start`, else the core engine's. A plugin
 *  that throws must not swallow the click, so the core engine starts instead. */
export function useVoiceStart(coreStart: () => void): () => void {
  const plugin = useSelectedPluginVoiceEngine()

  if (!plugin) {
    return coreStart
  }

  return () => {
    try {
      plugin.start()
    } catch (error) {
      notifyError(error, plugin.shortLabel)
      coreStart()
    }
  }
}

/**
 * Which engine the next voice conversation mounts: the chained
 * speech-to-text → Hermes → speech loop, or GPT-Live delegating to Hermes.
 *
 * Radio rows, not a toggle: the user is choosing between two named things and
 * the checked row tells them which one the next press starts. Rendered inside
 * whichever menu the layout has room for (the folded voice menu, or the
 * right-click menu on the start button), so the same rows appear in both.
 * Hidden while the backend has not answered or predates the mode, so we never
 * offer a switch the gateway would refuse with 4002. Plugin engines
 * (`composer.voiceEngines`) follow the core rows; picking one is a
 * renderer-side choice, picking a core row clears it.
 */
export function VoiceEngineRows({ disabled }: { disabled: boolean }) {
  const { t } = useI18n()
  const c = t.composer
  const status = useStore($voiceLiveStatus)
  const engines = useComposerVoiceEngines()
  const plugin = useSelectedPluginVoiceEngine()

  if (status === null) {
    return null
  }

  const liveAvailable = status.available

  return (
    <>
      <DropdownMenuLabel>{c.voiceEngine}</DropdownMenuLabel>
      <DropdownMenuRadioGroup
        onValueChange={value => {
          if (value.startsWith(PLUGIN_PREFIX)) {
            triggerHaptic('open')
            choosePluginVoiceEngine(value.slice(PLUGIN_PREFIX.length))

            return
          }

          if (value !== 'chained' && value !== 'gpt-live') {
            return
          }

          triggerHaptic('open')
          choosePluginVoiceEngine(null)
          setVoiceChatMode(value).catch(error => notifyError(error, c.voiceEngineChangeFailed))
        }}
        value={plugin ? `${PLUGIN_PREFIX}${plugin.id}` : selectedVoiceChatMode(status)}
      >
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled} value="chained">
          {c.voiceEngineChained}
        </DropdownMenuRadioItem>
        <DropdownMenuRadioItem className={dropdownMenuRow} disabled={disabled || !liveAvailable} value="gpt-live">
          <span className="flex min-w-0 flex-col">
            <span>{c.voiceEngineLive}</span>
            {liveAvailable ? null : (
              <span className="text-muted-foreground truncate text-xs">
                {status.reason ?? c.voiceEngineLiveNeedsKey}
              </span>
            )}
          </span>
        </DropdownMenuRadioItem>
        {engines.map(engine => (
          <DropdownMenuRadioItem
            className={dropdownMenuRow}
            disabled={disabled}
            key={engine.id}
            value={`${PLUGIN_PREFIX}${engine.id}`}
          >
            <span className="flex min-w-0 flex-col">
              <span>{engine.label}</span>
              {engine.detail ? <span className="text-muted-foreground truncate text-xs">{engine.detail}</span> : null}
            </span>
          </DropdownMenuRadioItem>
        ))}
      </DropdownMenuRadioGroup>
    </>
  )
}

/** Short engine name for tooltips, or null until the backend has answered. */
export function useVoiceEngineName(): null | string {
  const { t } = useI18n()
  const status = useStore($voiceLiveStatus)
  const plugin = useSelectedPluginVoiceEngine()

  if (status === null) {
    return null
  }

  if (plugin) {
    return plugin.shortLabel
  }

  return selectedVoiceChatMode(status) === 'gpt-live'
    ? t.composer.voiceEngineLiveShort
    : t.composer.voiceEngineChainedShort
}
