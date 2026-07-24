import { useEffect, useState } from 'react'

import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'
import { Mic, Plus } from '@/lib/icons'
import type { HermesConfigRecord } from '@/types/hermes'

import {
  addSttCommandProvider,
  type SttCommandFormat,
  type SttCommandProviderDraft,
  type SttCommandProviderErrors,
  validateSttCommandProvider
} from './helpers'

const EMPTY_DRAFT: SttCommandProviderDraft = {
  name: '',
  command: '',
  format: 'txt',
  language: '',
  model: ''
}

const FORMATS: SttCommandFormat[] = ['txt', 'json', 'srt', 'vtt']

export function VoiceCommandProviderAction({
  config,
  onApply
}: {
  config: HermesConfigRecord
  onApply: (config: HermesConfigRecord) => void
}) {
  const [open, setOpen] = useState(false)

  return (
    <>
      <div className="mb-3 flex justify-end">
        <Button onClick={() => setOpen(true)} size="sm" type="button" variant="outline">
          <Plus />
          Add local STT provider
        </Button>
      </div>
      <VoiceCommandProviderDialog config={config} onApply={onApply} onOpenChange={setOpen} open={open} />
    </>
  )
}

export function VoiceCommandProviderDialog({
  config,
  open,
  onOpenChange,
  onApply
}: {
  config: HermesConfigRecord
  open: boolean
  onOpenChange: (open: boolean) => void
  onApply: (config: HermesConfigRecord) => void
}) {
  const [draft, setDraft] = useState<SttCommandProviderDraft>(EMPTY_DRAFT)
  const [errors, setErrors] = useState<SttCommandProviderErrors>({})

  useEffect(() => {
    if (open) {
      setDraft(EMPTY_DRAFT)
      setErrors({})
    }
  }, [open])

  const update = <K extends keyof SttCommandProviderDraft>(key: K, value: SttCommandProviderDraft[K]) => {
    setDraft(current => ({ ...current, [key]: value }))

    if (key === 'name' || key === 'command') {
      setErrors(current => ({ ...current, [key]: undefined }))
    }
  }

  const addProvider = () => {
    const nextErrors = validateSttCommandProvider(config, draft)

    if (Object.keys(nextErrors).length > 0) {
      setErrors(nextErrors)

      return
    }

    onApply(addSttCommandProvider(config, draft))
    onOpenChange(false)
  }

  return (
    <Dialog onOpenChange={onOpenChange} open={open}>
      <DialogContent className="max-w-xl">
        <DialogHeader>
          <DialogTitle icon={Mic}>Add local STT provider</DialogTitle>
          <DialogDescription>
            Run a trusted local command for speech recognition. Hermes supplies the audio path and reads the transcript
            your command writes; the command runs with your full user permissions.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3">
          <div className="grid gap-1.5 text-xs text-muted-foreground">
            <label htmlFor="stt-provider-name">Provider name</label>
            <Input
              aria-describedby={errors.name ? 'stt-provider-name-error' : undefined}
              aria-invalid={Boolean(errors.name)}
              autoFocus
              id="stt-provider-name"
              onChange={event => update('name', event.target.value)}
              placeholder="sensevoice"
              value={draft.name}
            />
            {errors.name ? (
              <span className="text-[0.6875rem] leading-4 text-destructive" id="stt-provider-name-error">
                {errors.name}
              </span>
            ) : null}
          </div>

          <div className="grid gap-1.5 text-xs text-muted-foreground">
            <label htmlFor="stt-provider-command">Command</label>
            <Textarea
              aria-describedby={errors.command ? 'stt-provider-command-error' : 'stt-provider-command-help'}
              aria-invalid={Boolean(errors.command)}
              className="min-h-20 font-mono"
              id="stt-provider-command"
              onChange={event => update('command', event.target.value)}
              placeholder="sensevoice-cli {input_path} --output {output_path}"
              value={draft.command}
            />
            {errors.command ? (
              <span className="text-[0.6875rem] leading-4 text-destructive" id="stt-provider-command-error">
                {errors.command}
              </span>
            ) : (
              <span className="text-[0.6875rem] leading-4" id="stt-provider-command-help">
                Required: {'{input_path}'} and either {'{output_path}'} or {'{output_dir}'}.
              </span>
            )}
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="grid gap-1.5 text-xs text-muted-foreground">
              Language
              <Input
                onChange={event => update('language', event.target.value)}
                placeholder="Optional"
                value={draft.language}
              />
            </label>
            <label className="grid gap-1.5 text-xs text-muted-foreground">
              Model
              <Input
                onChange={event => update('model', event.target.value)}
                placeholder="Optional"
                value={draft.model}
              />
            </label>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <label className="grid gap-1.5 text-xs text-muted-foreground">
              Transcript format
              <Select onValueChange={value => update('format', value as SttCommandFormat)} value={draft.format}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {FORMATS.map(format => (
                    <SelectItem key={format} value={format}>
                      {format.toUpperCase()}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </label>
            <label className="grid gap-1.5 text-xs text-muted-foreground">
              Timeout
              <Input
                min={1}
                onChange={event =>
                  update('timeout', event.target.value === '' ? undefined : Number(event.target.value))
                }
                placeholder="300 seconds"
                type="number"
                value={draft.timeout ?? ''}
              />
            </label>
          </div>
        </div>

        <DialogFooter>
          <DialogClose asChild>
            <Button size="sm" type="button" variant="ghost">
              Cancel
            </Button>
          </DialogClose>
          <Button onClick={addProvider} size="sm" type="button">
            <Plus />
            Add provider
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
