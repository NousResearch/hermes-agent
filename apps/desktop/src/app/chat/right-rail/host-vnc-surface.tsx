import { useStore } from '@nanostores/react'
import { useEffect, useState } from 'react'

import { Codicon } from '@/components/ui/codicon'
import { getHostDisplay, type HostDisplayResponse } from '@/hermes'
import { useI18n } from '@/i18n'
import { $previewProfileKey } from '@/store/preview-profile'
import { $connection } from '@/store/session'

interface HostDisplayState {
  error: string | null
  loading: boolean
  response: HostDisplayResponse | null
}

const initialState: HostDisplayState = { error: null, loading: true, response: null }

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message.trim()) {
    return error.message
  }

  return String(error || 'Host VNC discovery failed')
}

export function HostVncSurface() {
  const { t } = useI18n()
  const previewProfileKey = useStore($previewProfileKey)
  const connection = useStore($connection)
  const connectionKey = `${connection?.mode || 'local'}:${connection?.profile || ''}:${connection?.baseUrl || ''}`
  const [reloadRequest, setReloadRequest] = useState(0)
  const [state, setState] = useState<HostDisplayState>(initialState)

  useEffect(() => {
    let cancelled = false

    setState(initialState)
    void getHostDisplay().then(
      response => {
        if (!cancelled) {
          setState({ error: null, loading: false, response })
        }
      },
      error => {
        if (!cancelled) {
          setState({ error: errorMessage(error), loading: false, response: null })
        }
      }
    )

    return () => {
      cancelled = true
    }
  }, [connectionKey, previewProfileKey, reloadRequest])

  const configuredUrl = state.response?.available ? state.response.url : null

  const sameOriginBlocked = (() => {
    if (!configuredUrl || !['http:', 'https:'].includes(window.location.protocol)) {
      return false
    }

    try {
      return new URL(configuredUrl).origin === window.location.origin
    } catch {
      return true
    }
  })()

  const url = sameOriginBlocked ? null : configuredUrl

  if (url) {
    return (
      <iframe
        allow="clipboard-read; clipboard-write; fullscreen"
        allowFullScreen
        className="h-full min-h-0 w-full border-0 bg-black"
        referrerPolicy="no-referrer"
        sandbox="allow-forms allow-same-origin allow-scripts"
        src={url}
        title={t.rightSidebar.hostVnc}
      />
    )
  }

  const message = state.loading
    ? t.preview.opening
    : sameOriginBlocked
      ? t.rightSidebar.hostVncSameOrigin
      : (state.error ?? state.response?.reason ?? t.preview.unavailable)

  return (
    <div className="grid h-full min-h-0 place-items-center bg-(--ui-editor-surface-background) p-6 text-center">
      <div className="max-w-sm rounded-xl border border-(--ui-stroke-tertiary) bg-(--ui-sidebar-surface-background) p-5 shadow-lg">
        <Codicon
          className={`mx-auto mb-3 text-(--ui-text-tertiary)${state.loading ? ' animate-spin' : ''}`}
          name={state.loading ? 'loading' : 'remote-explorer'}
          size="1.5rem"
        />
        <h2 className="text-sm font-semibold text-foreground">{t.rightSidebar.hostVnc}</h2>
        <p aria-live="polite" className="mt-2 text-xs leading-relaxed text-(--ui-text-tertiary)" role="status">
          {message}
        </p>
        {!state.loading && (
          <button
            className="mt-4 rounded-md border border-(--ui-stroke-secondary) px-3 py-1.5 text-xs font-semibold text-foreground hover:border-(--theme-primary) hover:bg-(--ui-control-hover-background) focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring"
            onClick={() => setReloadRequest(value => value + 1)}
            type="button"
          >
            {t.rightSidebar.tryAgain}
          </button>
        )}
      </div>
    </div>
  )
}
