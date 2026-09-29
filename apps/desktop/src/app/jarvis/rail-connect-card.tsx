import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { useNavigate } from 'react-router'

import { CONNECTION_STATUS_KEY, getConnectionStatus } from '@/api/connections'
import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { Plus, Sparkles } from '@/lib/icons'
import { cn } from '@/lib/utils'
import { requestComposerPrefill } from '@/store/composer'

import { GoogleConnectDialog } from '../connections/google-connect-dialog'
import { navigateToWorkspacePage, NEW_CHAT_ROUTE } from '../routes'

import { CONNECTION_ICONS } from './connection-icons'
import { JARVIS_CONNECTIONS, type JarvisConnection, suggestConnections } from './connections-catalog'
import { readJarvisOnboardingState } from './onboarding-state'
import { RailCard } from './rail-cards'

const COPY = {
  en: { connect: 'Connect', hint: 'Czesiek can do more once it can reach:', title: 'Connect more' },
  pl: { connect: 'Połącz', hint: 'Czesiek pomoże więcej, gdy będzie miał dostęp do:', title: 'Połącz więcej' }
} as const

/**
 * The next connections worth making, offered where the work happens instead of
 * only on a settings page: the ones chosen in setup first, then the broadly
 * useful. Reads presence only, and disappears once there is nothing to suggest.
 */
export function JarvisConnectCard({ connected }: { connected: boolean }) {
  const { locale, t } = useI18n()
  const copy = locale === 'pl' ? COPY.pl : COPY.en
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [googleOpen, setGoogleOpen] = useState(false)

  const status = useQuery({
    enabled: connected,
    queryFn: () => getConnectionStatus(),
    queryKey: CONNECTION_STATUS_KEY,
    staleTime: 30_000
  })

  const chosen = readJarvisOnboardingState()?.selections?.connections ?? []
  const suggestions = status.data ? suggestConnections(status.data, chosen) : []

  if (suggestions.length === 0) {
    return null
  }

  const start = (connection: JarvisConnection) => {
    const setup = connection.setup

    if (setup.kind === 'wizard') {
      setGoogleOpen(true)
    } else if (setup.kind === 'agent') {
      requestComposerPrefill(t.jarvisConnections.entries[connection.id].prompt)
      navigate(NEW_CHAT_ROUTE)
    } else {
      navigateToWorkspacePage(navigate, setup.route)
    }
  }

  return (
    <RailCard icon={Plus} testId="connect" title={copy.title}>
      <p className="mb-2 text-xs text-(--ui-text-tertiary)">{copy.hint}</p>
      <ul className="grid grid-cols-1 gap-1.5">
        {suggestions.map(id => {
          const connection = JARVIS_CONNECTIONS.find(item => item.id === id)!
          const { icon: Icon, tile } = CONNECTION_ICONS[id]
          const entry = t.jarvisConnections.entries[id]

          return (
            <li className="jarvis-well flex items-center gap-3 p-2" data-connection-suggestion={id} key={id}>
              <span className={cn('grid size-8 shrink-0 place-items-center rounded-lg', tile)}>
                <Icon className="size-4" />
              </span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm font-medium text-(--ui-text-primary)">{entry.name}</span>
                <span className="block truncate text-xs text-(--ui-text-tertiary)">{entry.description}</span>
              </span>
              <Button onClick={() => start(connection)} size="sm" type="button" variant="secondary">
                <Sparkles />
                {copy.connect}
              </Button>
            </li>
          )
        })}
      </ul>
      <GoogleConnectDialog
        onChanged={() => void queryClient.invalidateQueries({ queryKey: CONNECTION_STATUS_KEY })}
        onClose={() => setGoogleOpen(false)}
        open={googleOpen}
      />
    </RailCard>
  )
}
