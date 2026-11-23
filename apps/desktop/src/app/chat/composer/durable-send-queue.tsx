import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

import { navigateToWorkspacePage, sessionRoute } from '@/app/routes'
import { draftKey, isFreshDraftScope } from '@/store/composer'
import {
  $durableSends,
  durableComposerOwner,
  refreshDurableSends,
  startDurableSends,
  warmDurableOwner
} from '@/store/durable-composer'
import { setSessionOwnerHint } from '@/store/session'

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
  const entries = useStore($durableSends)
  const scope = draftKey(sessionKey)
  const owner = durableComposerOwner(scope)
  const [error, setError] = useState('')
  useEffect(() => {
    startDurableSends()
    void warmDurableOwner(scope)
  }, [scope])

  const visible = entries.filter(
    entry =>
      owner &&
      ['connectionId', 'profile', 'gatewayId', 'principalId'].every(
        field => entry.intent.owner[field as keyof typeof owner] === owner[field as keyof typeof owner]
      ) &&
      (entry.sessionId === scope ||
        ('sessionId' in entry.intent.target
          ? entry.intent.target.sessionId === scope
          : entry.intent.target.draftId === scope || isFreshDraftScope(scope)))
  )

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
              {labels[entry.state]}
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
