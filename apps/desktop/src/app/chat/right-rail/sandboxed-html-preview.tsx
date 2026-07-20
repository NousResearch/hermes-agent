import { useEffect, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { useI18n } from '@/i18n'
import { desktopFsCacheKey, readDesktopFileText } from '@/lib/desktop-fs'
import { Loader2, Lock } from '@/lib/icons'
import type { PreviewTarget } from '@/store/preview'

import { filePathForTarget, PreviewEmptyState } from './preview-file'
import {
  approveSandboxedHtml,
  buildSandboxedHtmlDocument,
  isSandboxedHtmlApproved,
  SANDBOXED_HTML_MAX_BYTES,
  SANDBOXED_HTML_PERMISSIONS,
  sandboxedHtmlApprovalIdentity,
  sha256Text
} from './sandboxed-html-approval'

const SAFE_BOOTSTRAP_DOCUMENT =
  '<!doctype html><html><head><meta http-equiv="Content-Security-Policy" content="default-src \'none\'"></head><body></body></html>'

const FRAME_NAME_PREFIX = 'hermes-sandboxed-html:'

interface ReadyPreview {
  approved: boolean
  digest: string
  identity: string
  path: string
  sandboxedDocument: string
  source: string
}

type PreviewState =
  | { status: 'error'; body: string; title: string }
  | { status: 'loading' }
  | ({ status: 'ready' } & ReadyPreview)

function frameName(): string {
  const id = crypto.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`

  return `${FRAME_NAME_PREFIX}${id}`
}

function ApprovedSandbox({ documentSource, onIsolationError }: { documentSource: string; onIsolationError: () => void }) {
  const [name] = useState(frameName)
  const [registered, setRegistered] = useState(false)
  const registrationPendingRef = useRef(false)

  useEffect(
    () => () => {
      void window.hermesDesktop?.sandboxedHtml?.unregisterFrame(name)
    },
    [name]
  )

  const registerFrame = () => {
    if (registered || registrationPendingRef.current) {
      return
    }

    const bridge = window.hermesDesktop?.sandboxedHtml

    if (!bridge) {
      onIsolationError()

      return
    }

    registrationPendingRef.current = true
    void bridge
      .registerFrame(name)
      .then(ok => {
        if (!ok) {
          onIsolationError()

          return
        }

        setRegistered(true)
      })
      .catch(onIsolationError)
  }

  return (
    <iframe
      allow={SANDBOXED_HTML_PERMISSIONS}
      className="h-full w-full border-0 bg-background"
      name={name}
      onLoad={registerFrame}
      referrerPolicy="no-referrer"
      sandbox="allow-scripts"
      srcDoc={registered ? documentSource : SAFE_BOOTSTRAP_DOCUMENT}
      title="Sandboxed HTML preview"
    />
  )
}

export function SandboxedHtmlPreview({ reloadKey, target }: { reloadKey: number; target: PreviewTarget }) {
  const { t } = useI18n()
  const copy = t.preview.sandboxedHtml
  const [loadNonce, setLoadNonce] = useState(0)
  const [showSource, setShowSource] = useState(false)
  const [state, setState] = useState<PreviewState>({ status: 'loading' })

  useEffect(() => {
    let active = true
    const requestedPath = filePathForTarget(target)
    const connectionKey = desktopFsCacheKey()

    setState({ status: 'loading' })
    setShowSource(false)

    void readDesktopFileText(requestedPath)
      .then(async result => {
        if (result.binary) {
          throw new Error(copy.binaryBody)
        }

        if (result.truncated) {
          throw new Error(copy.truncatedBody)
        }

        if (!result.path || typeof result.path !== 'string' || typeof result.text !== 'string') {
          throw new Error(copy.invalidBody)
        }

        const actualBytes = new TextEncoder().encode(result.text).byteLength

        if ((result.byteSize ?? actualBytes) > SANDBOXED_HTML_MAX_BYTES || actualBytes > SANDBOXED_HTML_MAX_BYTES) {
          throw new Error(copy.tooLargeBody)
        }

        const digest = await sha256Text(result.text)
        const identity = sandboxedHtmlApprovalIdentity(connectionKey, result.path)

        if (!active) {
          return
        }

        setState({
          approved: isSandboxedHtmlApproved(identity, digest),
          digest,
          identity,
          path: result.path,
          sandboxedDocument: buildSandboxedHtmlDocument(result.text),
          source: result.text,
          status: 'ready'
        })
      })
      .catch(error => {
        if (!active) {
          return
        }

        setState({
          body: error instanceof Error ? error.message : copy.readFailedBody,
          status: 'error',
          title: copy.unavailableTitle
        })
      })

    return () => {
      active = false
    }
  }, [copy, loadNonce, reloadKey, target])

  if (state.status === 'loading') {
    return (
      <div className="absolute inset-0 grid place-items-center bg-background text-muted-foreground">
        <div className="flex items-center gap-2 text-xs">
          <Loader2 className="size-4 animate-spin" />
          {copy.loading}
        </div>
      </div>
    )
  }

  if (state.status === 'error') {
    return (
      <PreviewEmptyState
        body={state.body}
        primaryAction={{ label: copy.tryAgain, onClick: () => setLoadNonce(value => value + 1) }}
        title={state.title}
        tone="warning"
      />
    )
  }

  if (state.approved) {
    return (
      <ApprovedSandbox
        documentSource={state.sandboxedDocument}
        onIsolationError={() =>
          setState({ body: copy.isolationBody, status: 'error', title: copy.isolationTitle })
        }
      />
    )
  }

  return (
    <div className="absolute inset-0 overflow-auto bg-background px-6 py-8">
      <div className="mx-auto grid max-w-2xl gap-5">
        <div className="flex items-start gap-3 border-b border-border/60 pb-5">
          <div className="mt-0.5 grid size-8 shrink-0 place-items-center rounded-[4px] bg-muted text-muted-foreground">
            <Lock className="size-4" />
          </div>
          <div className="min-w-0 space-y-1">
            <h2 className="text-sm font-medium text-foreground">{copy.approvalTitle}</h2>
            <p className="text-xs leading-relaxed text-muted-foreground">{copy.approvalBody}</p>
          </div>
        </div>

        <dl className="grid gap-3 text-xs">
          <div className="grid gap-1 sm:grid-cols-[7rem_minmax(0,1fr)]">
            <dt className="font-medium text-foreground">{copy.fileLabel}</dt>
            <dd className="break-all font-mono text-muted-foreground">{state.path}</dd>
          </div>
          <div className="grid gap-1 sm:grid-cols-[7rem_minmax(0,1fr)]">
            <dt className="font-medium text-foreground">{copy.digestLabel}</dt>
            <dd className="font-mono text-muted-foreground">sha256:{state.digest.slice(0, 12)}</dd>
          </div>
          <div className="grid gap-1 sm:grid-cols-[7rem_minmax(0,1fr)]">
            <dt className="font-medium text-foreground">{copy.permissionsLabel}</dt>
            <dd className="leading-relaxed text-muted-foreground">{copy.permissionsBody}</dd>
          </div>
        </dl>

        <div className="flex flex-wrap items-center gap-3 border-t border-border/60 pt-5">
          <Button
            onClick={() => {
              approveSandboxedHtml(state.identity, state.digest)
              setState({ ...state, approved: true })
            }}
            type="button"
          >
            {copy.runAction}
          </Button>
          <Button onClick={() => setShowSource(value => !value)} size="inline" type="button" variant="text">
            {showSource ? copy.hideSource : copy.showSource}
          </Button>
        </div>

        {showSource && (
          <pre className="max-h-80 overflow-auto border border-border/60 bg-muted/30 p-3 text-left font-mono text-[0.6875rem] leading-5 text-foreground">
            <code>{state.source}</code>
          </pre>
        )}
      </div>
    </div>
  )
}
