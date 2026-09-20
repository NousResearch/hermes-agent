import { useEffect, useState } from 'react'

import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { deviceLabel, listVoiceDevices, VOICE_INPUT_DEVICE_KEY } from '@/lib/voice-devices'

import { CONTROL_TEXT, EMPTY_SELECT_VALUE } from './constants'

/**
 * Microphone / speaker picker (part of the local GUI patch).
 *
 * Enumerated at open time and re-enumerated on `devicechange`, so plugging in a headset updates the
 * list without a reload. Two things it is careful about:
 *
 *  - **A configured device that is no longer connected is called out** rather than silently replaced
 *    by the default — the surprise this setting exists to remove. Recording falls back to the
 *    default in that case (see `voice-devices.ts`), but the row says so.
 *  - Labels are empty until the page has been granted microphone access, so entries fall back to
 *    "Microphone N" / "Speaker N" instead of rendering blank rows.
 */
export function VoiceDevicePicker({
  schemaKey,
  value,
  onChange
}: {
  schemaKey: string
  value: string
  onChange: (next: string) => void
}) {
  const kind = schemaKey === VOICE_INPUT_DEVICE_KEY ? 'input' : 'output'
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [loaded, setLoaded] = useState(false)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      const lists = await listVoiceDevices()

      if (!cancelled) {
        setDevices(kind === 'input' ? lists.inputs : lists.outputs)
        setLoaded(true)
      }
    }

    void load()

    const media = typeof navigator === 'undefined' ? undefined : navigator.mediaDevices
    media?.addEventListener?.('devicechange', load)

    return () => {
      cancelled = true
      media?.removeEventListener?.('devicechange', load)
    }
  }, [kind])

  const current = value.trim()
  const missing = Boolean(current) && loaded && !devices.some(device => device.deviceId === current)

  return (
    <div className="flex flex-col items-end gap-1">
      <Select
        onValueChange={next => onChange(next === EMPTY_SELECT_VALUE ? '' : next)}
        value={current || EMPTY_SELECT_VALUE}
      >
        <SelectTrigger className={CONTROL_TEXT}>
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value={EMPTY_SELECT_VALUE}>System default</SelectItem>
          {devices.map((device, index) => (
            <SelectItem
              key={device.deviceId || index}
              value={device.deviceId || EMPTY_SELECT_VALUE}
            >
              {deviceLabel(device, index, kind)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {missing ? (
        <span className="max-w-56 text-right text-xs text-muted-foreground">
          That device is not connected — the system default is used until it is back.
        </span>
      ) : null}
      {loaded && devices.length === 0 ? (
        <span className="max-w-56 text-right text-xs text-muted-foreground">
          No devices found. Grant microphone access, then reopen this page.
        </span>
      ) : null}
    </div>
  )
}
