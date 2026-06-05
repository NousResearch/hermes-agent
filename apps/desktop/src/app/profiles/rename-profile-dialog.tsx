import { useEffect, useState } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { renameProfile } from '@/hermes'
import { AlertTriangle } from '@/lib/icons'
import { cn } from '@/lib/utils'

import { isValidProfileName, PROFILE_NAME_HINT } from './create-profile-dialog'

// Self-contained rename (owns the renameProfile call) so every caller just
// reacts via onRenamed. Unchanged name is a no-op close.
export function RenameProfileDialog({
  currentName,
  onClose,
  onRenamed,
  open
}: {
  currentName: string
  onClose: () => void
  onRenamed?: (name: string) => Promise<void> | void
  open: boolean
}) {
  const [name, setName] = useState(currentName)
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)

  useEffect(() => {
    if (!open) {
      return
    }

    setName(currentName)
    setError(null)
    setStatus('idle')
  }, [currentName, open])

  const trimmed = name.trim()
  const unchanged = trimmed === currentName
  const invalid = trimmed !== '' && !unchanged && !isValidProfileName(trimmed)
  const busy = status === 'saving' || status === 'done'

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    if (unchanged) {
      onClose()

      return
    }

    if (!trimmed || invalid) {
      setError(invalid ? `Invalid name. ${PROFILE_NAME_HINT}` : 'Name is required.')

      return
    }

    setStatus('saving')
    setError(null)

    try {
      await renameProfile(currentName, trimmed)
      await onRenamed?.(trimmed)
      setStatus('done')
      window.setTimeout(onClose, 800)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : 'Failed to rename profile')
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Rename profile</DialogTitle>
          <DialogDescription>
            Renaming updates the profile directory and any wrapper scripts in{' '}
            <span className="font-mono">~/.local/bin</span>.
          </DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={handleSubmit}>
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="rename-profile-name">
              New name
            </label>
            <Input
              aria-invalid={invalid}
              autoFocus
              id="rename-profile-name"
              onChange={event => setName(event.target.value)}
              value={name}
            />
            <p className={cn('text-[0.66rem] leading-4', invalid ? 'text-destructive' : 'text-muted-foreground')}>
              {PROFILE_NAME_HINT}
            </p>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter>
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              Cancel
            </Button>
            <Button disabled={busy || invalid || unchanged} type="submit">
              <ActionStatus busy="Renaming…" done="Renamed" idle="Rename" state={status} />
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
