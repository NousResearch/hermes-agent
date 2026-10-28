import { type ProfileScope } from '@/api/client'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { confirm } from '@/store/confirm'

import { useModelPresets } from './use-model-presets'
interface ModelPresetsProps {
  scope: ProfileScope
}

/** Snapshot operations belong to the plugin; live-chat changes use the existing slash dispatcher. */
export function ModelPresets({ scope }: ModelPresetsProps) {
  const {
    t,
    p,
    query,
    state,
    choice,
    canApply,
    busy,
    output,
    outputTarget,
    setSelected,
    editing,
    setEditing,
    name,
    setName,
    routeKey,
    dialogRoute,
    setDialogRoute,
    mutate,
    apply
  } = useModelPresets(scope)

  if (!state) {
    return (
      <div className="mb-4 text-xs text-(--ui-text-tertiary)" role="status">
        {query.isError ? p.failed : query.isPending ? t.common.loading : p.unavailable}
        {query.isError && (
          <Button onClick={() => void query.refetch()} size="xs" variant="text">
            {t.common.retry}
          </Button>
        )}
      </div>
    )
  }

  return (
    <div aria-label={p.title} className="mb-4 grid gap-2">
      <div className="flex flex-wrap items-center gap-2">
        <Select
          disabled={busy || state.names.length === 0}
          onValueChange={value => {
            setSelected(value)
            void mutate({ action: 'use', name: value })
          }}
          value={choice}
        >
          <SelectTrigger aria-label={p.title}>
            <SelectValue placeholder={p.custom} />
          </SelectTrigger>
          <SelectContent>
            {state.names.map(value => (
              <SelectItem key={value} value={value}>
                {value}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <Button
          disabled={busy}
          onClick={() => {
            setName('')
            setDialogRoute(routeKey)
            setEditing('save')
          }}
          size="sm"
          variant="secondary"
        >
          {p.save}
        </Button>
        <Button
          disabled={busy || !choice}
          onClick={() => {
            setName(choice)
            setDialogRoute(routeKey)
            setEditing('rename')
          }}
          size="sm"
          variant="text"
        >
          {p.rename}
        </Button>
        <Button
          disabled={busy || !choice}
          onClick={async () => {
            if (
              await confirm({ title: p.delete, description: p.deleteHint, confirmLabel: p.delete, destructive: true })
            ) {
              await mutate({ action: 'delete', name: choice })
            }
          }}
          size="sm"
          variant="text"
        >
          {p.delete}
        </Button>
      </div>
      <p className="text-xs text-(--ui-text-tertiary)">
        {p.hint} {state.active ? `${p.next}: ${state.active}` : p.custom}
      </p>
      <div className="flex flex-wrap gap-2">
        <Button
          disabled={busy || !choice}
          onClick={() => void mutate({ action: 'use', name: choice })}
          size="sm"
          variant="secondary"
        >
          {p.next}
        </Button>
        <Button disabled={busy || !choice || !canApply} onClick={() => void apply()} size="sm" variant="outline">
          {p.apply}
        </Button>
      </div>
      {output?.target === outputTarget && (
        <p className="text-xs text-(--ui-text-secondary)" role="status">
          {output.text}
        </p>
      )}
      {editing && dialogRoute === routeKey && (
        <Dialog
          onOpenChange={open => {
            if (!open && !busy) {
              setEditing(null)
            }
          }}
          open
        >
          <DialogContent>
            <DialogHeader>
              <DialogTitle>{editing === 'save' ? p.save : p.rename}</DialogTitle>
              <DialogDescription>{p.nameHint}</DialogDescription>
            </DialogHeader>
            <form
              onSubmit={event => {
                event.preventDefault()

                if (dialogRoute !== routeKey) {
                  return
                }

                void mutate({
                  action: editing,
                  name: editing === 'rename' ? choice : name,
                  ...(editing === 'rename' ? { new_name: name } : {})
                })
              }}
            >
              <Input
                aria-label={p.name}
                autoFocus
                disabled={busy}
                maxLength={64}
                onChange={event => setName(event.target.value)}
                pattern="[A-Za-z0-9][A-Za-z0-9_.-]{0,63}"
                required
                value={name}
              />
              <DialogFooter>
                <Button disabled={busy} onClick={() => setEditing(null)} type="button" variant="text">
                  {t.common.cancel}
                </Button>
                <Button disabled={busy || !name} type="submit">
                  {editing === 'save' ? p.save : p.rename}
                </Button>
              </DialogFooter>
            </form>
          </DialogContent>
        </Dialog>
      )}
    </div>
  )
}
