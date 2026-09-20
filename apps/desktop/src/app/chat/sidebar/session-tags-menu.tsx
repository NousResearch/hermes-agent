import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import type { MenuKit } from '@/components/ui/actions-menu'
import { ContextMenuCheckboxItem } from '@/components/ui/context-menu'
import { DropdownMenuCheckboxItem } from '@/components/ui/dropdown-menu'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { requestGatewayForAgent } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { refreshProjectTree } from '@/store/projects'
import { $connection, $cronSessions, $messagingSessions, $sessions, sessionMatchesStoredId } from '@/store/session'
import { invalidateSessionTags } from '@/store/session-tags'
import { $archivedSessions } from '@/store/sidebar-archive'

export function SessionTagsMenu({
  kit,
  sessionId,
  profile,
  tags,
  connectionId: requestedConnectionId
}: {
  kit: MenuKit
  sessionId: string
  profile?: string
  tags?: string[]
  connectionId?: string | null
}) {
  const { t } = useI18n()
  const sessions = useStore($sessions)
  const cron = useStore($cronSessions)
  const messaging = useStore($messagingSessions)
  const connection = useStore($connection)
  const archived = useStore($archivedSessions)
  const owner = profile || connection?.profile || 'default'
  const connectionId = requestedConnectionId === undefined ? (connection?.connectionId ?? null) : requestedConnectionId

  const row = [...sessions, ...cron, ...messaging, ...archived].find(
    s =>
      sessionMatchesStoredId(s, sessionId) &&
      (s.profile || 'default') === owner &&
      (s.connection_id ?? connection?.connectionId ?? null) === connectionId
  )

  const [catalogue, setCatalogue] = useState<string[]>([])
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [updatedTags, setUpdatedTags] = useState<string[] | null>(null)
  const assigned = updatedTags ?? row?.tags ?? tags ?? []
  useEffect(() => setUpdatedTags(null), [connectionId, owner, sessionId])
  useEffect(() => {
    let alive = true
    setCatalogue([])
    setError('')
    requestGatewayForAgent<{ tags: string[] }>(connectionId, owner, 'session.tags.list', { profile: owner })
      .then(result => {
        if (alive) {
          setCatalogue(result.tags)
        }
      })
      .catch(() => {
        if (alive) {
          setError(t.sidebar.tags.loadError)
        }
      })

    return () => {
      alive = false
    }
  }, [connectionId, owner, t.sidebar.tags.loadError])

  async function setTag(tag: string, value: boolean) {
    if (busy || !tag.trim()) {
      return
    }

    setBusy(true)

    try {
      const result = await requestGatewayForAgent<{ tags: string[] }>(connectionId, owner, 'session.tags.set', {
        profile: owner,
        session_id: row?.id || sessionId,
        tag: tag.trim(),
        assigned: value
      })

      for (const store of [$sessions, $cronSessions, $messagingSessions, $archivedSessions]) {
        store.set(
          store
            .get()
            .map(s =>
              s.id === (row?.id || sessionId) &&
              (s.profile || 'default') === owner &&
              (s.connection_id || connection?.connectionId || null) === connectionId
                ? { ...s, tags: result.tags }
                : s
            )
        )
      }

      invalidateSessionTags()
      setUpdatedTags(result.tags)
      void refreshProjectTree()
      setCatalogue(current => [...new Set([...current, ...result.tags])].sort())
      setName('')
    } catch (err) {
      notifyError(err, t.sidebar.tags.updateError)
    } finally {
      setBusy(false)
    }
  }

  const Checkbox = kit.copyAppearance === 'context-menu-item' ? ContextMenuCheckboxItem : DropdownMenuCheckboxItem

  return (
    <>
      {error && <kit.Item disabled>{error}</kit.Item>}
      {[...new Set([...catalogue, ...assigned])].sort().map(tag => (
        <Checkbox
          checked={assigned.includes(tag)}
          disabled={busy}
          key={tag}
          onSelect={event => {
            event.preventDefault()
            void setTag(tag, !assigned.includes(tag))
          }}
        >
          {tag}
        </Checkbox>
      ))}
      <kit.Separator />
      <form
        className="p-2"
        onKeyDown={event => event.stopPropagation()}
        onSubmit={event => {
          event.preventDefault()
          void setTag(name, true)
        }}
      >
        <Input
          aria-label={t.sidebar.tags.newTag}
          disabled={busy}
          onChange={event => setName(event.target.value)}
          placeholder={t.sidebar.tags.newTag}
          value={name}
        />
        <button className="mt-2 text-xs" disabled={busy || !name.trim()} type="submit">
          {t.sidebar.tags.createAndAssign}
        </button>
      </form>
    </>
  )
}
