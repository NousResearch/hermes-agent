import { useState } from 'react'

import { Button } from '@/components/ui/button'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import type { CronJob } from '@/hermes'
import { useI18n } from '@/i18n'

import { useAgentSettingsCopy } from './copy'
import { type AgentSettingsOwner, saveAgentRoutine } from './data'

interface RoutineDialogProps {
  owner: AgentSettingsOwner
  job?: CronJob
  onClose: () => void
  onSaved: () => void
}

export function RoutineDialog({ owner, job, onClose, onSaved }: RoutineDialogProps) {
  const { t } = useI18n()
  const c = useAgentSettingsCopy()
  const initialSchedule = job?.schedule?.expr || job?.schedule_display || job?.schedule?.display || ''
  const [name, setName] = useState(job?.name || '')
  const [schedule, setSchedule] = useState(initialSchedule)
  const [prompt, setPrompt] = useState(job?.prompt || '')
  const [enabled, setEnabled] = useState(job?.enabled ?? true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const script = Boolean(job?.script || job?.no_agent)

  async function save(event: React.FormEvent) {
    event.preventDefault()

    if (saving) {return}
    setSaving(true)
    setError('')

    try {
      const values = job
        ? {
            name: name.trim(),
            enabled,
            ...(schedule !== initialSchedule ? { schedule: schedule.trim() } : {}),
            ...(!script ? { prompt: prompt.trim() } : {})
          }
        : { name: name.trim(), schedule: schedule.trim(), prompt: prompt.trim(), deliver: 'local' }

      await saveAgentRoutine(owner, values, job?.id)
      onSaved()
      onClose()
    } catch (err) {
      setError(err instanceof Error ? err.message : t.errors.genericFailure)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog
      onOpenChange={open => {
        if (!open) {onClose()}
      }}
      open
    >
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{job ? c.routines : c.addRoutine}</DialogTitle>
          <DialogDescription>
            {owner.profile}
            {!job && ` · ${c.localDelivery}`}
          </DialogDescription>
        </DialogHeader>
        <form className="grid gap-4" onSubmit={save}>
          <label className="grid gap-1 text-xs">
            {c.name}
            <Input onChange={event => setName(event.target.value)} value={name} />
          </label>
          <label className="grid gap-1 text-xs">
            {c.schedule}
            <Input
              onChange={event => setSchedule(event.target.value)}
              placeholder="0 9 * * *"
              required
              value={schedule}
            />
          </label>
          {script ? (
            <p className="text-xs text-muted-foreground">{c.scriptRoutine}</p>
          ) : (
            <label className="grid gap-1 text-xs">
              {c.prompt}
              <Textarea onChange={event => setPrompt(event.target.value)} required value={prompt} />
            </label>
          )}
          {job && (
            <label className="flex items-center gap-2 text-xs">
              <Checkbox checked={enabled} onCheckedChange={value => setEnabled(value === true)} />
              {c.enabled}
            </label>
          )}
          {error && (
            <p className="text-xs text-destructive" role="alert">
              {error}
            </p>
          )}
          <DialogFooter>
            <Button onClick={onClose} type="button" variant="secondary">
              {t.common.cancel}
            </Button>
            <Button disabled={saving} type="submit">
              {t.common.save}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
