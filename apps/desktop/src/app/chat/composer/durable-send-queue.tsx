import { useStore } from '@nanostores/react'
import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router'

import { navigateToWorkspacePage, sessionRoute } from '@/app/routes'
import { usePaneVisible } from '@/components/pane-shell/pane-visibility'
import { announceNewSessionDraftKey, draftKey, freshDraftScope, isFreshDraftScope, migrateSessionDraft, requestComposerDraftSync } from '@/store/composer'
import {
  $durableSends,
  durableComposerOwner,
  refreshDurableSends,
  startDurableSends,
  warmDurableOwner
} from '@/store/durable-composer'
import { adoptDraftPreviewTabs } from '@/store/preview'
import { setFreshDraftReady, setNewChatWorkspaceTarget, setSessionOwnerHint } from '@/store/session'

import { useComposerScope } from './scope'

const labels = {
  staging: 'Salvando anexos…',
  queued: 'Na fila — salvo neste dispositivo',
  sending: 'Enviando…',
  waiting: 'Aguardando conexão — salvo neste dispositivo',
  blocked: 'Envio bloqueado — cópia preservada',
  confirmed: 'Recebido pelo servidor',
  cancelled: 'Cancelado'
}

export function DurableSendQueue({ sessionKey }: { sessionKey: string | null }) {
  const navigate = useNavigate()
  const paneVisible = usePaneVisible()
  const composerTarget = useComposerScope().target
  const entries = useStore($durableSends)
  const scope = draftKey(sessionKey)
  const owner = durableComposerOwner(scope)
  const [error, setError] = useState('')
  useEffect(() => {
    startDurableSends()
    void warmDurableOwner(scope)
  }, [scope])

  const visible = useMemo(() => entries.filter(
    entry =>
      owner &&
      ['connectionId', 'profile', 'gatewayId', 'principalId'].every(
        field => entry.intent.owner[field as keyof typeof owner] === owner[field as keyof typeof owner]
      ) &&
      (entry.sessionId === scope ||
        ('sessionId' in entry.intent.target
          ? entry.intent.target.sessionId === scope
          : entry.intent.target.draftId === scope || isFreshDraftScope(scope)))
  ), [entries, owner, scope])

  useEffect(() => {
    if (!paneVisible || !isFreshDraftScope(scope) || freshDraftScope() !== scope) {return}

    const created = visible.find(entry => entry.sessionId && entry.state !== 'cancelled' &&
      !('sessionId' in entry.intent.target) && entry.intent.target.draftId === scope)

    if (!created?.sessionId) {return}
    // Only the origin's still-visible fresh lifecycle follows this create. Another fresh
    // draft or an existing chat must never be switched by a late queue acknowledgement.
    const stored = created.sessionId
    setSessionOwnerHint(stored, { connectionId: created.intent.owner.connectionId, profile: created.intent.owner.profile })
    requestComposerDraftSync('flush', composerTarget)
    migrateSessionDraft(scope, stored)
    announceNewSessionDraftKey(stored)
    adoptDraftPreviewTabs(stored)
    setFreshDraftReady(false)
    setNewChatWorkspaceTarget(undefined)
    navigateToWorkspacePage(navigate, sessionRoute(stored))
  }, [composerTarget, navigate, paneVisible, scope, visible])

  if (!visible.length) {
    return null
  }

  return (
    <div aria-label="Mensagens preservadas" className="max-h-40 overflow-auto text-xs">
      {visible.map(entry => (
        <div className="flex items-center gap-2 py-1" key={entry.intent.id} role="status">
          <div className="min-w-0 flex-1">
            <div className="truncate">
              {entry.intent.text || 'Mensagem com anexos'}
              {entry.intent.files.length ? ` · ${entry.intent.files.length} anexo(s)` : ''}
            </div>
            <div className="text-muted-foreground">
              {entry.cancelPending ? 'Cancelamento aguardando confirmação — cópia preservada' : labels[entry.state]}
              {entry.problem === 'uncertain' ? ' — confirmação incerta; não será reenviado automaticamente' : ''}
            </div>
          </div>
          {!entry.submitStarted && !['sending', 'confirmed', 'cancelled'].includes(entry.state) && (
            <button
              onClick={() => {
                void window
                  .hermesDesktop!.durableSend!.cancel(entry.intent.id)
                  .then(refreshDurableSends)
                  .catch(() => setError('O envio já está em andamento; aguarde a confirmação.'))
              }}
              type="button"
            >
              Cancelar
            </button>
          )}
          {entry.sessionId && (
            <button
              onClick={() => {
                setSessionOwnerHint(entry.sessionId!, {
                  connectionId: entry.intent.owner.connectionId,
                  profile: entry.intent.owner.profile
                })
                navigateToWorkspacePage(navigate, sessionRoute(entry.sessionId!))
              }}
              type="button"
            >
              Abrir conversa
            </button>
          )}
        </div>
      ))}
      {error && <div role="alert">{error}</div>}
    </div>
  )
}
