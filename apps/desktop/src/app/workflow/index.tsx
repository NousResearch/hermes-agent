import { useCallback, useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'

const LANGFLOW_URL = 'http://127.0.0.1:7860'

type WorkflowWebview = HTMLElement & {
  getURL?: () => string
  reloadIgnoringCache?: () => void
}

interface LoadError {
  code?: number
  description?: string
  url?: string
}

export function WorkflowView() {
  const { t } = useI18n()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const webviewRef = useRef<WorkflowWebview | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<LoadError | null>(null)
  const [mountKey, setMountKey] = useState(0)

  const reload = useCallback(() => {
    setLoadError(null)
    setLoading(true)

    if (webviewRef.current?.reloadIgnoringCache) {
      webviewRef.current.reloadIgnoringCache()

      return
    }

    setMountKey(key => key + 1)
  }, [])

  const openInBrowser = useCallback(() => {
    void window.hermesDesktop?.openExternal?.(LANGFLOW_URL).catch(() => undefined)
  }, [])

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    host.replaceChildren()
    webviewRef.current = null
    setLoadError(null)
    setLoading(true)

    const webview = document.createElement('webview') as WorkflowWebview
    webview.className = 'flex h-full w-full flex-1 bg-background'
    webview.setAttribute('partition', 'persist:hermes-workflow')
    webview.setAttribute('src', LANGFLOW_URL)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const onFail = (event: Event) => {
      const detail = event as Event & {
        errorCode?: number
        errorDescription?: string
        isMainFrame?: boolean
        validatedURL?: string
      }

      if (detail.errorCode === -3 || detail.isMainFrame === false) {
        return
      }

      setLoadError({
        code: detail.errorCode,
        description: detail.errorDescription,
        url: detail.validatedURL || webview.getURL?.() || LANGFLOW_URL
      })
      setLoading(false)
    }

    const onStart = () => {
      setLoadError(null)
      setLoading(true)
    }

    const onStop = () => setLoading(false)

    webview.addEventListener('did-fail-load', onFail)
    webview.addEventListener('did-start-loading', onStart)
    webview.addEventListener('did-stop-loading', onStop)
    host.appendChild(webview)
    webviewRef.current = webview

    return () => {
      webview.removeEventListener('did-fail-load', onFail)
      webview.removeEventListener('did-start-loading', onStart)
      webview.removeEventListener('did-stop-loading', onStop)
      webview.remove()
    }
  }, [mountKey])

  return (
    <section aria-label={t.workflow.title} className="relative flex h-full min-h-0 flex-1 overflow-hidden bg-background">
      <div className="h-full min-h-0 flex-1" ref={hostRef} />

      {loading && !loadError && (
        <div className="pointer-events-none absolute inset-0 grid place-items-center bg-background/80">
          <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
            <Codicon className="text-sm" name="loading" spinning />
            <span>{t.workflow.loading}</span>
          </div>
        </div>
      )}

      {loadError && (
        <div className="absolute inset-0 grid place-items-center bg-background px-6" role="alert">
          <div className="flex w-full max-w-md flex-col items-center gap-4 text-center">
            <div className="grid size-10 place-items-center rounded-md border border-border/70 bg-muted/40 text-muted-foreground">
              <Codicon className="text-lg" name="debug-disconnect" />
            </div>
            <div className="space-y-1.5">
              <h1 className="text-sm font-semibold text-foreground">{t.workflow.backendUnavailable}</h1>
              <p className="text-xs leading-5 text-muted-foreground">
                {t.workflow.backendUnavailableDetail(loadError.url || LANGFLOW_URL)}
              </p>
              {loadError.description && (
                <p className="text-[0.6875rem] leading-4 text-muted-foreground/80">{loadError.description}</p>
              )}
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              <Button onClick={reload} size="sm" type="button" variant="secondary">
                <Codicon name="refresh" />
                {t.common.refresh}
              </Button>
              <Button onClick={openInBrowser} size="sm" type="button" variant="outline">
                <Codicon name="link-external" />
                {t.workflow.openInBrowser}
              </Button>
            </div>
          </div>
        </div>
      )}
    </section>
  )
}
