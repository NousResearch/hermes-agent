import { useStore } from '@nanostores/react'
import { type FormEvent, useEffect, useState } from 'react'

import {
  browserBack,
  browserForward,
  browserNavigate,
  browserReload,
  BrowserSlot,
  captureBrowserTab,
  saveBrowserCapture
} from '@/app/browser/persistent'
import {
  $browserCapture,
  $browserState,
  addBrowserTab,
  BrowserTabLimitError,
  clearBrowserCapture,
  closeBrowserTab,
  normalizeBrowserRuntimeUrl,
  setBrowserActiveTab,
  setBrowserQcOpen,
  toggleBrowserTabPin
} from '@/app/browser/store'
import { Button } from '@/components/ui/button'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useI18n } from '@/i18n'
import { Check, ChevronLeft, ChevronRight, ExternalLink, Globe, Pin, Plus, RefreshCw, Save, X } from '@/lib/icons'
import { notifyError } from '@/store/notifications'

const isInlineImage = (value: string) => /^data:image\/(?:png|jpe?g|webp|gif|avif|bmp)(?:;[^,]*)?,/i.test(value)

const validBrowserUrl = (value: string) => {
  const normalized = normalizeBrowserRuntimeUrl(value)

  return Boolean(normalized && normalized !== 'about:blank')
}

const externalBrowserUrl = (value: string) => /^https?:/i.test(value)

export function BrowserPane() {
  const { t } = useI18n()
  const copy = t.desktop

  const browserState = useStore($browserState)
  const capture = useStore($browserCapture)
  const activeTab = browserState.tabs.find(tab => tab.id === browserState.activeTabId) ?? null
  const activeTabId = activeTab?.id ?? null
  const activeTabUrl = activeTab?.url ?? ''
  const [url, setUrl] = useState(activeTabUrl)
  const [invalidUrl, setInvalidUrl] = useState(false)

  useEffect(() => {
    setUrl(isInlineImage(activeTabUrl) ? '' : activeTabUrl)
    setInvalidUrl(false)
  }, [activeTabId, activeTabUrl])

  const navigate = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    if (!activeTab) {
      return
    }

    const target = normalizeBrowserRuntimeUrl(url)

    if (!validBrowserUrl(url)) {
      setInvalidUrl(true)

      return
    }

    setInvalidUrl(false)
    browserNavigate(activeTab.id, target)
  }

  const captureActiveTab = async () => {
    if (!activeTab) {
      return
    }

    try {
      await captureBrowserTab(activeTab.id)
    } catch (error) {
      notifyError(error, copy.imagePreviewFailed)
    }
  }

  const addTab = () => {
    try {
      addBrowserTab()
    } catch (error) {
      if (error instanceof BrowserTabLimitError) {
        notifyError(undefined, copy.browserTabLimit)

        return
      }

      notifyError(error, t.common.failed)
    }
  }

  return (
    <section
      aria-label={copy.browserTitle}
      className="flex h-full min-h-0 min-w-0 flex-1 flex-col bg-(--ui-editor-surface-background)"
      data-testid="browser-pane"
    >
      <header className="flex shrink-0 flex-col gap-2 border-b border-(--ui-stroke-tertiary) p-2">
        <div className="flex min-w-0 items-center gap-1">
          <Tabs onValueChange={setBrowserActiveTab} value={browserState.activeTabId ?? undefined}>
            <TabsList className="max-w-full justify-start overflow-x-auto">
              {browserState.tabs.map(tab => (
                <div className="flex items-center" key={tab.id}>
                  <TabsTrigger value={tab.id}>
                    <span className="max-w-32 truncate">{tab.title || tab.url || copy.browserNewTab}</span>
                  </TabsTrigger>
                  <Button
                    aria-label={copy.browserCloseTab}
                    onClick={() => closeBrowserTab(tab.id)}
                    size="icon-xs"
                    variant="ghost"
                  >
                    <X />
                  </Button>
                </div>
              ))}
            </TabsList>
          </Tabs>
          <Button aria-label={copy.browserNewTab} onClick={addTab} size="icon-xs" variant="ghost">
            <Plus />
          </Button>
          <Button
            aria-label={copy.browserQc}
            onClick={() => setBrowserQcOpen(true)}
            size="icon-xs"
            variant="ghost"
          >
            <Check />
          </Button>
        </div>
        <div className="flex min-w-0 items-center gap-1">
          <Button
            aria-label={copy.browserBack}
            disabled={!activeTab}
            onClick={() => activeTab && browserBack(activeTab.id)}
            size="icon-xs"
            variant="ghost"
          >
            <ChevronLeft />
          </Button>
          <Button
            aria-label={copy.browserForward}
            disabled={!activeTab}
            onClick={() => activeTab && browserForward(activeTab.id)}
            size="icon-xs"
            variant="ghost"
          >
            <ChevronRight />
          </Button>
          <Button
            aria-label={copy.browserReload}
            disabled={!activeTab}
            onClick={() => activeTab && browserReload(activeTab.id)}
            size="icon-xs"
            variant="ghost"
          >
            <RefreshCw />
          </Button>
          <form className="min-w-0 flex-1" onSubmit={navigate}>
            <Input
              aria-invalid={invalidUrl}
              aria-label={copy.browserUrlPlaceholder}
              onChange={event => setUrl(event.target.value)}
              placeholder={copy.browserUrlPlaceholder}
              value={url}
            />
          </form>
          <Button
            aria-label={activeTab?.pinned ? copy.browserUnpin : copy.browserPin}
            aria-pressed={activeTab?.pinned}
            disabled={!activeTab}
            onClick={() => activeTab && toggleBrowserTabPin(activeTab.id)}
            size="icon-xs"
            variant="ghost"
          >
            <Pin />
          </Button>
          <Button
            aria-label={copy.browserOpenExternal}
            disabled={!activeTab || !externalBrowserUrl(activeTab.url)}
            onClick={() => activeTab && void window.hermesDesktop?.openExternal?.(activeTab.url)}
            size="icon-xs"
            variant="ghost"
          >
            <ExternalLink />
          </Button>
          <Button
            aria-label={copy.browserCapture}
            disabled={!activeTab?.url}
            onClick={() => void captureActiveTab()}
            size="icon-xs"
            variant="ghost"
          >
            <Globe />
          </Button>
        </div>
        {invalidUrl ? <p className="text-xs text-destructive">{copy.browserInvalidUrl}</p> : null}
      </header>
      <div className="relative flex min-h-0 min-w-0 flex-1">
        <BrowserSlot />
        {!activeTab?.url ? (
          <p className="pointer-events-none absolute inset-0 grid place-items-center text-sm text-(--ui-text-tertiary)">
            {copy.browserEmpty}
          </p>
        ) : null}
      </div>
      <Dialog onOpenChange={open => !open && clearBrowserCapture()} open={Boolean(capture)}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle>{copy.browserCapturePreview}</DialogTitle>
            <DialogDescription>{copy.browserCapturePreview}</DialogDescription>
          </DialogHeader>
          {capture ? (
            <img
              alt={copy.browserCapturePreview}
              className="max-h-[60vh] w-full object-contain"
              src={capture.dataUrl}
            />
          ) : null}
          <div className="flex justify-end gap-2">
            <Button
              disabled={!capture?.captureId}
              onClick={() => {
                if (!capture) {
                  return
                }

                void saveBrowserCapture(capture.captureId)
                  .then(result => {
                    if (!result.canceled) {
                      clearBrowserCapture()
                    }
                  })
                  .catch(error => notifyError(error, copy.imageDownloadFailed))
              }}
              size="sm"
              variant="secondary"
            >
              <Save />
              {copy.browserSaveCapture}
            </Button>
            <Button onClick={() => clearBrowserCapture()} size="sm" variant="ghost">
              <X />
              {copy.browserClosePreview}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </section>
  )
}
