import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useI18n } from '@/i18n'
import { type AudioInputDevice, displayAudioInputLabel, isVirtualAudioLabel } from '@/lib/audio-input'
import {
  $audioInputDeviceId,
  listAudioInputDevices,
  setAudioInputDeviceId,
  SYSTEM_DEFAULT_AUDIO_INPUT
} from '@/store/audio-input'

import { ListRow } from './primitives'

export function MicrophoneSetting() {
  const { t } = useI18n()
  const copy = t.composer.microphone
  const deviceId = useStore($audioInputDeviceId)
  const [devices, setDevices] = useState<AudioInputDevice[]>([])

  useEffect(() => {
    let cancelled = false

    void (async () => {
      await window.hermesDesktop?.requestMicrophoneAccess?.()
      const next = await listAudioInputDevices()

      if (!cancelled) {
        setDevices(next)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [])

  const value = deviceId ?? SYSTEM_DEFAULT_AUDIO_INPUT

  return (
    <ListRow
      action={
        <Select onValueChange={next => setAudioInputDeviceId(next)} value={value}>
          <SelectTrigger aria-label={copy.title} className="w-full min-w-56" size="sm">
            <SelectValue placeholder={copy.systemDefault}>
              {displayAudioInputLabel(value, devices, copy.systemDefault)}
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={SYSTEM_DEFAULT_AUDIO_INPUT}>{copy.systemDefault}</SelectItem>
            {devices
              .filter(device => device.deviceId !== SYSTEM_DEFAULT_AUDIO_INPUT)
              .map(device => (
                <SelectItem key={device.deviceId} value={device.deviceId}>
                  {device.label}
                  {isVirtualAudioLabel(device.label) ? ` — ${copy.virtualHint}` : ''}
                </SelectItem>
              ))}
          </SelectContent>
        </Select>
      }
      description={copy.description}
      title={copy.title}
    />
  )
}
