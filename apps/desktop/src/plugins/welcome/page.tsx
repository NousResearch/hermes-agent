/**
 * Welcome dashboard page — the first thing a user sees when opening Unitalk.
 * Three sections: greeting with prompt input, recent conversations, info card.
 */

import './welcome.css'

import {
  type ChangeEvent,
  type KeyboardEvent,
  useCallback,
  useState
} from 'react'

import {
  host,
  useQuery,
  useValue
} from '@hermes/plugin-sdk'

// ── helpers ──────────────────────────────────────────────────────────────────

const SUGGESTIONS = [
  'Rédiger un email',
  'Analyser un rapport',
  'Préparer une présentation',
  'Résumer un document'
] as const

interface SessionRow {
  id: string
  title: string
  preview: string
  /** Unix ms timestamp. */
  updated: number
}

/** Fetch recent sessions — the same list the sidebar shows, last 5. */
async function fetchRecentSessions(): Promise<SessionRow[]> {
  try {
    const raw = await window.fetch('/api/sessions?limit=5')
    if (!raw.ok) {
      return []
    }
    const data = await raw.json()
    const rows: unknown[] = Array.isArray(data?.sessions) ? data.sessions : Array.isArray(data) ? data : []
    return rows.slice(0, 5).map((row: any) => ({
      id: String(row.id ?? ''),
      title: String(row.title ?? 'Sans titre').trim() || 'Sans titre',
      preview: String(row.preview ?? row.last_message_preview ?? '').slice(0, 80).trim(),
      updated: row.updated ?? row.updated_at ?? row.created ?? Date.now()
    }))
  } catch {
    return []
  }
}

/** Simple relative time string: "il y a 2h", "il y a 5m", "à l'instant". */
function relativeTime(ms: number): string {
  const now = Date.now()
  const diff = now - ms
  if (diff < 60_000) return "à l'instant"
  const mins = Math.floor(diff / 60_000)
  if (mins < 60) return `il y a ${mins}min`
  const hours = Math.floor(mins / 60)
  if (hours < 24) return `il y a ${hours}h`
  const days = Math.floor(hours / 24)
  return `il y a ${days}j`
}

// ── components ───────────────────────────────────────────────────────────────

interface RecentConversationsProps {
  sessions: SessionRow[]
  loading: boolean
  onSelect: (id: string) => void
}

function RecentConversations({ sessions, loading, onSelect }: RecentConversationsProps) {
  if (loading) {
    return (
      <div className="welcome-sessions-card">
        <h2>Conversations récentes</h2>
        <div className="welcome-empty">Chargement…</div>
      </div>
    )
  }

  return (
    <div className="welcome-sessions-card">
      <h2>Conversations récentes</h2>
      {sessions.length === 0 ? (
        <div className="welcome-empty">
          Aucune conversation. Commencez par confier une mission à votre Collaborateur.
        </div>
      ) : (
        sessions.map(session => (
          <div
            key={session.id}
            className="welcome-session-item"
            onClick={() => onSelect(session.id)}
            role="button"
            tabIndex={0}
            onKeyDown={(e: KeyboardEvent) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelect(session.id)
              }
            }}
          >
            <span className="welcome-session-title">{session.title}</span>
            {session.preview ? (
              <span className="welcome-session-preview">{session.preview}</span>
            ) : null}
            <span className="welcome-session-time">
              {relativeTime(session.updated)}
            </span>
          </div>
        ))
      )}
    </div>
  )
}

function InfoCard() {
  const model = useValue(host.state.model)
  const profile = useValue(host.state.profile)

  return (
    <div className="welcome-info-card">
      <h2>Votre espace</h2>
      <div className="welcome-info-row">
        <span className="welcome-info-dot" />
        <span>
          Connecté via <strong>{profile || 'Default'}</strong>
        </span>
      </div>
      {model ? (
        <div className="welcome-info-row">
          <span className="welcome-info-label">Modèle :</span>
          <span>{model}</span>
        </div>
      ) : null}
      <div className="welcome-info-row" style={{ marginTop: 8 }}>
        <span className="welcome-info-label">Crédits inclus</span>
      </div>
    </div>
  )
}

// ── page ─────────────────────────────────────────────────────────────────────

export function WelcomePage() {
  const [prompt, setPrompt] = useState('')

  const { data: sessions = [], isLoading } = useQuery<SessionRow[]>({
    queryFn: fetchRecentSessions,
    queryKey: ['welcome', 'recent-sessions'],
    refetchInterval: 60_000,
    staleTime: 30_000
  })

  const handleSubmit = useCallback(() => {
    const trimmed = prompt.trim()
    if (!trimmed) {
      return
    }
    // Open a fresh draft — the composer takes it from here.
    host.navigate('/')
    // Write the prompt into the composer's backing store.
    // The composer listens on this atom. Hack but effective.
    try {
      const $draft = (window as any).__HERMES_COMPOSER_DRAFT__
      if ($draft?.set) {
        $draft.set(trimmed)
      }
    } catch {
      // Fallback: just navigate; user can type it themselves.
    }
    setPrompt('')
  }, [prompt])

  const handleChipClick = useCallback((text: string) => {
    setPrompt(text)
    // Defer so the state update lands before we try to submit.
    setTimeout(() => {
      try {
        const $draft = (window as any).__HERMES_COMPOSER_DRAFT__
        if ($draft?.set) {
          $draft.set(text)
        }
      } catch {
        // navigate anyway
      }
      host.navigate('/')
    }, 50)
  }, [])

  const handleSessionSelect = useCallback((id: string) => {
    if (typeof host.openSession === 'function') {
      host.openSession(id, { awaitHydration: false }).catch(() => {
        host.navigate('/')
      })
    } else {
      host.navigate('/')
    }
  }, [])

  return (
    <div className="welcome-page">
      {/* ── Greeting ── */}
      <div className="welcome-greeting">
        <h1>Bonjour, je suis votre Collaborateur IA</h1>
        <p className="welcome-greeting-subtitle">
          Je suis prêt à vous aider. Que voulez-vous faire&nbsp;?
        </p>
        <textarea
          className="welcome-input"
          rows={2}
          placeholder="Décrivez votre besoin en quelques mots…"
          value={prompt}
          onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setPrompt(e.target.value)}
          onKeyDown={(e: KeyboardEvent<HTMLTextAreaElement>) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault()
              handleSubmit()
            }
          }}
        />
        <div className="welcome-chips">
          {SUGGESTIONS.map(suggestion => (
            <button
              key={suggestion}
              className="welcome-chip"
              type="button"
              onClick={() => handleChipClick(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>

      {/* ── Two-column section ── */}
      <div className="welcome-columns">
        <RecentConversations
          sessions={sessions}
          loading={isLoading}
          onSelect={handleSessionSelect}
        />
        <InfoCard />
      </div>
    </div>
  )
}