import { useEffect, useState } from 'react'

import { ActionStatus } from '@/components/ui/action-status'
import { Button } from '@/components/ui/button'
import { ConfirmDialog } from '@/components/ui/confirm-dialog'
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
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import {
  generateProviderId,
  type CustomProviderApiMode,
  type CustomProviderEntry
} from '@/lib/custom-provider-config'

function isValidHttpUrl(value: string): boolean {
  try {
    const url = new URL(value)
    return url.protocol === 'http:' || url.protocol === 'https:'
  } catch {
    return false
  }
}

export function ProviderFormDialog({
  existingNames = [],
  initial = null,
  mode = 'custom',
  onClose,
  onDelete,
  onSave,
  onSaveBuiltIn,
  open,
  providerName = '',
  redactedApiKey
}: {
  /** Raw names of OTHER custom providers, for uniqueness checks. */
  existingNames?: string[]
  /** When set, the dialog is in edit mode and prefilled. */
  initial?: CustomProviderEntry | null
  /** 'custom' → full provider form; 'builtin' → credentials-only form. */
  mode?: 'builtin' | 'custom'
  onClose: () => void
  onDelete?: (name: string) => Promise<void> | void
  onSave: (entry: CustomProviderEntry) => Promise<void> | void
  /** Built-in mode: persist API key (+ optional base URL override) via setEnvVar. */
  onSaveBuiltIn?: (apiKey: string, baseUrl?: string) => Promise<void> | void
  open: boolean
  /** Display name for built-in providers (read-only in the dialog). */
  providerName?: string
  /** Redacted API key (e.g. "sk-…abc") shown as placeholder so the user knows
   *  a key is already saved. Fetched from getEnvVars() by the parent. */
  redactedApiKey?: string
}) {
  const { t } = useI18n()
  const p = t.providerManager
  const isBuiltin = mode === 'builtin'
  const isEdit = initial !== null

  const [name, setName] = useState('')
  const [baseUrl, setBaseUrl] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [apiMode, setApiMode] = useState<CustomProviderApiMode>('chat_completions')
  const [status, setStatus] = useState<'done' | 'idle' | 'saving'>('idle')
  const [error, setError] = useState<null | string>(null)
  const [confirmDelete, setConfirmDelete] = useState(false)

  useEffect(() => {
    if (!open) {
      return
    }

    setName(initial?.name ?? '')
    setBaseUrl(initial?.base_url ?? '')
    setApiKey('')
    setApiMode(initial?.api_mode ?? 'chat_completions')
    setError(null)
    setStatus('idle')
    setConfirmDelete(false)
  }, [open, initial])

  const trimmedName = name.trim()
  // The stored identity is the generated id (custom:<id>), derived from the
  // friendly name and uniquified against existing providers. Collisions are
  // resolved with a numeric suffix instead of blocking the user.
  const generatedId = trimmedName === '' ? '' : generateProviderId(trimmedName, existingNames)
  const urlInvalid = baseUrl.trim() !== '' && !isValidHttpUrl(baseUrl.trim())
  const busy = status === 'saving' || status === 'done'

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault()

    // Built-in mode: persist credentials via setEnvVar. Base URL is optional.
    if (isBuiltin) {
      if (baseUrl.trim() && urlInvalid) {
        setError(p.invalidUrl)
        return
      }
      setStatus('saving')
      setError(null)
      try {
        await onSaveBuiltIn?.(apiKey, baseUrl.trim() || undefined)
        setStatus('done')
        window.setTimeout(onClose, 600)
      } catch (err) {
        setStatus('idle')
        setError(err instanceof Error ? err.message : p.invalidUrl)
      }
      return
    }

    if (!baseUrl.trim() || urlInvalid) {
      setError(p.invalidUrl)
      return
    }

    setStatus('saving')
    setError(null)

    try {
      await onSave({
        name: generatedId,
        base_url: baseUrl.trim(),
        api_key: apiKey,
        api_mode: apiMode,
        models: []
      })
      setStatus('done')
      window.setTimeout(onClose, 600)
    } catch (err) {
      setStatus('idle')
      setError(err instanceof Error ? err.message : p.invalidUrl)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && !busy && onClose()} open={open}>
      <DialogContent
        className="max-w-md"
        onEscapeKeyDown={e => e.preventDefault()}
        onInteractOutside={e => e.preventDefault()}
        onPointerDownOutside={e => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle>{isBuiltin ? p.editProviderCredentials : isEdit ? p.editProvider : p.addProvider}</DialogTitle>
          <DialogDescription>{isBuiltin ? p.apiKeyDescription : p.baseUrlPlaceholder}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-4" onSubmit={handleSubmit}>
          {/* Provider name: editable for custom, read-only for built-in. */}
          {isBuiltin ? (
            <div className="grid gap-1.5">
              <span className="text-xs font-medium">{p.providerName}</span>
              <div className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) px-3 py-2 text-sm text-foreground">
                {providerName}
              </div>
            </div>
          ) : (
            <div className="grid gap-1.5">
              <label className="text-xs font-medium" htmlFor="provider-form-name">
                {p.providerName}
              </label>
              <Input
                autoFocus
                disabled={busy}
                id="provider-form-name"
                onChange={event => setName(event.target.value)}
                placeholder={p.providerNamePlaceholder}
                value={name}
              />
            </div>
          )}

          {/* Generated provider id: read-only preview of the stored identity
              (custom:<id>). Auto-derived from the friendly name and uniquified
              against existing providers. */}
          {!isBuiltin && generatedId !== '' && (
            <div className="grid gap-1.5">
              <span className="text-xs font-medium">{p.providerId}</span>
              <div
                aria-label={p.providerId}
                className="rounded-md border border-(--ui-stroke-tertiary) bg-(--ui-bg-tertiary) px-3 py-2 font-mono text-sm text-muted-foreground"
              >
                {generatedId}
              </div>
            </div>
          )}

          {/* Base URL: required for custom, optional override for built-in. */}
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="provider-form-base-url">
              {isBuiltin ? p.baseUrlOverride : p.baseUrl}
            </label>
            <Input
              aria-invalid={urlInvalid}
              disabled={busy}
              id="provider-form-base-url"
              onChange={event => setBaseUrl(event.target.value)}
              placeholder={isBuiltin ? p.baseUrlOverridePlaceholder : p.baseUrlPlaceholder}
              value={baseUrl}
            />
          </div>

          {/* API key: primary field for built-in, optional for custom.
              Built-in mode shows the already-saved key as a readable masked
              placeholder (same redacted_value the API Keys settings page shows)
              so the user knows a key is on file. The field is type="text" in
              that mode so the placeholder isn't masked to dots; custom mode
              stays type="password". */}
          <div className="grid gap-1.5">
            <label className="text-xs font-medium" htmlFor="provider-form-api-key">
              {p.apiKey}
            </label>
            <Input
              autoFocus={isBuiltin}
              disabled={busy}
              id="provider-form-api-key"
              onChange={event => setApiKey(event.target.value)}
              placeholder={isBuiltin && redactedApiKey ? redactedApiKey : p.apiKeyPlaceholder}
              type={isBuiltin ? 'text' : 'password'}
              value={apiKey}
            />
          </div>

          {/* API mode: custom providers only. */}
          {!isBuiltin && (
            <div className="grid gap-1.5">
              <label className="text-xs font-medium" htmlFor="provider-form-api-mode">
                {p.apiMode}
              </label>
              <Select
                disabled={busy}
                onValueChange={value => setApiMode(value as CustomProviderApiMode)}
                value={apiMode}
              >
                <SelectTrigger className="h-9 rounded-md" id="provider-form-api-mode">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="chat_completions">{p.apiModeChat}</SelectItem>
                  <SelectItem value="anthropic_messages">{p.apiModeAnthropic}</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          <DialogFooter className="gap-2">
            {!isBuiltin && isEdit && onDelete && initial && (
              <Button
                className="mr-auto"
                disabled={busy}
                onClick={() => setConfirmDelete(true)}
                type="button"
                variant="destructive"
              >
                {t.common.delete}
              </Button>
            )}
            <Button disabled={busy} onClick={onClose} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button
              disabled={isBuiltin ? busy || urlInvalid : busy || !trimmedName || !baseUrl.trim() || urlInvalid}
              type="submit"
            >
              <ActionStatus busy={t.common.saving} done={t.common.done} idle={t.common.save} state={status} />
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>

      {!isBuiltin && isEdit && onDelete && initial && (
        <ConfirmDialog
          cancelLabel={t.common.cancel}
          confirmLabel={t.common.delete}
          description={p.confirmDelete}
          destructive
          onClose={() => setConfirmDelete(false)}
          onConfirm={async () => {
            await onDelete(initial.name)
            onClose()
          }}
          open={confirmDelete}
          title={p.editProvider}
        />
      )}
    </Dialog>
  )
}
