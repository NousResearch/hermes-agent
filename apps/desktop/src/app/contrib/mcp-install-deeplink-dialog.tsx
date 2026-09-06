import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

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
import { getHermesConfigRecord, saveMcpServers } from '@/hermes'
import { useI18n } from '@/i18n'
import { AlertTriangle } from '@/lib/icons'
import { MCP_DEEPLINK_NAME_RE } from '@/lib/mcp-deeplink'
import { getServers } from '@/lib/mcp-servers'
import { $mcpInstallRequest } from '@/store/mcp-deeplink-install'
import { notify, readableError } from '@/store/notifications'

import { setHermesConfigCache } from '../hooks/use-config-record'

/**
 * Explicit-confirm gate for `hermes://mcp/install` deep links. The payload is
 * arbitrary attacker-controllable input (any web page can open the link), so
 * this dialog shows the server name and the FULL pretty-printed config —
 * exactly what would be written — and nothing touches config until the user
 * confirms. stdio (`command`) entries carry an extra caution banner because
 * confirming lets Hermes spawn that local process. An existing server name is
 * never silently overwritten: confirm stays blocked until the user picks a
 * fresh name or cancels.
 */
export function McpInstallDeepLinkDialog() {
  const { t } = useI18n()
  const m = t.settings.mcp
  const navigate = useNavigate()
  const request = useStore($mcpInstallRequest)

  const [name, setName] = useState('')
  const [existingNames, setExistingNames] = useState<null | string[]>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<null | string>(null)

  // (Re)arm per request: seed the editable name and fetch the current server
  // map so a same-name conflict is visible before the user confirms.
  useEffect(() => {
    if (!request) {
      return
    }

    setName(request.name)
    setExistingNames(null)
    setSaving(false)
    setError(null)

    let cancelled = false

    getHermesConfigRecord()
      .then(config => {
        if (!cancelled) {
          setExistingNames(Object.keys(getServers(config)))
        }
      })
      .catch(() => {
        // Conflict preflight failed (offline backend?) — confirm still re-fetches
        // and merges, so leave the dialog usable rather than wedging it.
        if (!cancelled) {
          setExistingNames([])
        }
      })

    return () => {
      cancelled = true
    }
  }, [request])

  if (!request) {
    return null
  }

  const trimmedName = name.trim()
  const nameValid = MCP_DEEPLINK_NAME_RE.test(trimmedName)
  const nameConflict = existingNames?.includes(trimmedName) ?? false
  const checkingConflicts = existingNames === null

  const close = () => {
    if (!saving) {
      $mcpInstallRequest.set(null)
    }
  }

  const confirm = async () => {
    if (saving || !nameValid || nameConflict || checkingConflicts) {
      return
    }

    setSaving(true)
    setError(null)

    try {
      // Merge over the FRESHEST server map — saveMcpServers replaces the whole
      // `mcp_servers` document, so saving over a stale snapshot would drop
      // servers added elsewhere since the dialog opened.
      const current = getServers(await getHermesConfigRecord())

      if (Object.prototype.hasOwnProperty.call(current, trimmedName)) {
        setExistingNames(Object.keys(current))
        setError(m.deepLinkNameConflict(trimmedName))

        return
      }

      const nextServers = { ...current, [trimmedName]: request.config }
      await saveMcpServers(nextServers)
      setHermesConfigCache(previous => (previous ? { ...previous, mcp_servers: nextServers } : previous))
      notify({ kind: 'success', title: m.savedTitle, message: m.savedMessage(trimmedName) })
      $mcpInstallRequest.set(null)
      navigate(`/skills?tab=mcp&server=${encodeURIComponent(trimmedName)}`)
    } catch (err) {
      setError(readableError(err, m.saveFailed).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog onOpenChange={value => !value && close()} open>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle>{m.deepLinkTitle}</DialogTitle>
          <DialogDescription>{m.deepLinkDescription}</DialogDescription>
        </DialogHeader>

        <div className="flex flex-col gap-3">
          {request.transport === 'stdio' && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{m.deepLinkStdioWarning}</span>
            </div>
          )}

          <label className="flex flex-col gap-1 text-xs text-muted-foreground">
            {m.name}
            <Input onChange={event => setName(event.target.value)} value={name} />
          </label>

          {!nameValid && <p className="text-xs text-destructive">{m.deepLinkNameInvalid}</p>}
          {nameValid && nameConflict && (
            <p className="text-xs text-destructive">{m.deepLinkNameConflict(trimmedName)}</p>
          )}

          <div className="flex flex-col gap-1 text-xs text-muted-foreground">
            {m.serverJson}
            <pre className="max-h-64 overflow-auto rounded-md border border-border bg-muted/40 p-2 font-mono text-xs whitespace-pre-wrap break-all text-foreground">
              {JSON.stringify(request.config, null, 2)}
            </pre>
          </div>

          {error && (
            <div className="flex items-start gap-2 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
              <AlertTriangle className="mt-0.5 size-3.5 shrink-0" />
              <span>{error}</span>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button disabled={saving} onClick={close} type="button" variant="ghost">
            {t.common.cancel}
          </Button>
          <Button
            disabled={saving || !nameValid || nameConflict || checkingConflicts}
            onClick={() => void confirm()}
            variant={request.transport === 'stdio' ? 'destructive' : 'default'}
          >
            {saving ? t.common.saving : m.deepLinkConfirm}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
