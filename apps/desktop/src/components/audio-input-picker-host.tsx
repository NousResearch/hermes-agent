import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { isVirtualAudioLabel } from '@/lib/audio-input'
import { Mic } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { $audioInputPickRequest, settleAudioInputPick, SYSTEM_DEFAULT_AUDIO_INPUT } from '@/store/audio-input'

export function AudioInputPickerHost() {
  const { t } = useI18n()
  const copy = t.composer.microphone
  const request = useStore($audioInputPickRequest)
  const [shown, setShown] = useState(request)
  const [selected, setSelected] = useState(SYSTEM_DEFAULT_AUDIO_INPUT)

  useEffect(() => {
    if (request) {
      setShown(request)

      const preferred =
        request.devices.find(device => !isVirtualAudioLabel(device.label) && device.deviceId !== 'default')?.deviceId ??
        SYSTEM_DEFAULT_AUDIO_INPUT

      setSelected(preferred)
    }
  }, [request])

  if (!shown) {
    return null
  }

  const choices = shown.devices.filter(
    (device, index, list) => list.findIndex(entry => entry.deviceId === device.deviceId) === index
  )

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {
          settleAudioInputPick(null)
        }
      }}
      open={request !== null}
    >
      <DialogContent className="min-w-80 sm:max-w-md" onOpenAutoFocus={event => event.preventDefault()}>
        <DialogHeader>
          <DialogTitle icon={Mic}>{copy.chooseTitle}</DialogTitle>
          <DialogDescription>{copy.chooseDescription}</DialogDescription>
        </DialogHeader>
        <div className="grid gap-1">
          <button
            className={cn(
              'w-full rounded-md px-3 py-2 text-left text-[length:var(--conversation-text-font-size)]',
              selected === SYSTEM_DEFAULT_AUDIO_INPUT
                ? 'bg-(--ui-bg-tertiary) text-foreground'
                : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
            )}
            onClick={() => setSelected(SYSTEM_DEFAULT_AUDIO_INPUT)}
            type="button"
          >
            {copy.systemDefault}
          </button>
          {choices
            .filter(device => device.deviceId !== SYSTEM_DEFAULT_AUDIO_INPUT)
            .map(device => (
              <button
                className={cn(
                  'w-full rounded-md px-3 py-2 text-left text-[length:var(--conversation-text-font-size)]',
                  selected === device.deviceId
                    ? 'bg-(--ui-bg-tertiary) text-foreground'
                    : 'text-(--ui-text-secondary) hover:bg-(--chrome-action-hover) hover:text-foreground'
                )}
                key={device.deviceId}
                onClick={() => setSelected(device.deviceId)}
                type="button"
              >
                <div>{device.label}</div>
                {isVirtualAudioLabel(device.label) ? (
                  <div className="mt-0.5 text-[length:var(--conversation-caption-font-size)] text-(--ui-text-tertiary)">
                    {copy.virtualHint}
                  </div>
                ) : null}
              </button>
            ))}
        </div>
        <DialogFooter>
          <Button onClick={() => settleAudioInputPick(null)} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button onClick={() => settleAudioInputPick(selected)} type="button">
            {copy.useThis}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
