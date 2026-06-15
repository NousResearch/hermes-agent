'use client'

import type { SyntaxHighlighterProps } from '@assistant-ui/react-streamdown'
import { type FC, memo, useState } from 'react'
import ShikiHighlighter from 'react-shiki'

import {
  CodeCard,
  CodeCardBody,
  CodeCardHeader,
  CodeCardIcon,
  CodeCardSubtitle,
  CodeCardTitle
} from '@/components/chat/code-card'
import { SyntaxHighlighter } from '@/components/chat/shiki-highlighter'
import { Codicon } from '@/components/ui/codicon'
import { CopyButton } from '@/components/ui/copy-button'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { useI18n } from '@/i18n'
import { cn } from '@/lib/utils'

/**
 * Renders ```html fences as a live, Claude-artifacts-style preview. Registered
 * per-language via `componentsByLanguage` on StreamdownTextPrimitive — the
 * global `SyntaxHighlighter` override replaces streamdown's built-in code
 * dispatcher (the only place its plugins are consulted), so a plugins-only
 * setup never runs; this component is the supported escape hatch (same wiring
 * as MermaidDiagram).
 *
 * While streaming (`defer`) or when the fence is empty we fall back to the
 * regular code card, so the user always at least sees the source build up.
 */
interface HtmlPreviewProps extends SyntaxHighlighterProps {
  defer?: boolean
}

// Mirror shiki-highlighter.tsx so the inline "Code" view matches the rest of
// the app (full react-shiki bundle, light-dark() theme follows color-scheme).
const SHIKI_THEME = { dark: 'github-dark-default', light: 'github-light-default' } as const

const SHIKI_COLOR_REPLACEMENTS: Record<string, Record<string, string>> = {
  'github-light-default': { '#6e7781': '#57606a' }
}

// SECURITY: `allow-scripts` only — never `allow-same-origin`. The HTML is
// untrusted LLM output running inside an Electron renderer; keeping the frame in
// an opaque origin denies it access to app cookies, storage, and the parent DOM.
// White background since HTML pages assume a white canvas. `reloadKey` re-mounts
// the frame so Refresh re-runs the document's scripts from a clean state.
const PreviewFrame: FC<{ className: string; reloadKey: number; source: string }> = ({
  className,
  reloadKey,
  source
}) => (
  <iframe
    className={className}
    key={reloadKey}
    sandbox="allow-scripts"
    srcDoc={source}
    title="html-preview"
  />
)

const HEADER_ICON_BUTTON =
  'flex size-5 items-center justify-center rounded-md text-muted-foreground opacity-55 transition-colors hover:bg-muted hover:text-foreground hover:opacity-100'

const HtmlPreviewImpl: FC<HtmlPreviewProps> = props => {
  const { code, defer = false } = props
  const { t } = useI18n()
  const [showCode, setShowCode] = useState(false)
  const [expanded, setExpanded] = useState(false)
  const [reloadKey, setReloadKey] = useState(0)
  const source = (code ?? '').trim()

  // Streaming or empty fence: keep the plain code card so the user watches the
  // markup arrive. We never preview a half-written document.
  if (defer || !source) {
    return <SyntaxHighlighter {...props} defer={defer} />
  }

  return (
    <>
      <CodeCard>
        <CodeCardHeader>
          <CodeCardTitle>
            <CodeCardIcon name="code" />
            {t.assistant.tool.code}
            <CodeCardSubtitle> · html</CodeCardSubtitle>
          </CodeCardTitle>
          <div className="flex items-center gap-1">
            <div className="flex items-center rounded-md border border-border p-0.5">
              <button
                aria-pressed={!showCode}
                className={cn(
                  'rounded-[0.25rem] px-1.5 py-0.5 text-[0.65rem] font-medium leading-none transition-colors',
                  showCode ? 'text-muted-foreground hover:text-foreground' : 'bg-muted text-foreground'
                )}
                onClick={() => setShowCode(false)}
                type="button"
              >
                {t.preview.renderedPreview}
              </button>
              <button
                aria-pressed={showCode}
                className={cn(
                  'rounded-[0.25rem] px-1.5 py-0.5 text-[0.65rem] font-medium leading-none transition-colors',
                  showCode ? 'bg-muted text-foreground' : 'text-muted-foreground hover:text-foreground'
                )}
                onClick={() => setShowCode(true)}
                type="button"
              >
                {t.assistant.tool.code}
              </button>
            </div>
            {!showCode && (
              <>
                <button
                  aria-label={t.common.refresh}
                  className={HEADER_ICON_BUTTON}
                  onClick={() => setReloadKey(key => key + 1)}
                  title={t.common.refresh}
                  type="button"
                >
                  <Codicon name="refresh" size={12} />
                </button>
                <button
                  aria-label={t.preview.expand}
                  className={HEADER_ICON_BUTTON}
                  onClick={() => setExpanded(true)}
                  title={t.preview.expand}
                  type="button"
                >
                  <Codicon name="screen-full" size={12} />
                </button>
              </>
            )}
            <CopyButton
              appearance="inline"
              className="-my-1 -mr-1 h-5 px-1 opacity-55 hover:opacity-100"
              iconClassName="size-2.5"
              label={t.assistant.tool.copyCode}
              showLabel={false}
              text={source}
            />
          </div>
        </CodeCardHeader>
        {showCode ? (
          <CodeCardBody>
            <pre className="aui-shiki m-0 overflow-hidden bg-transparent p-0">
              <ShikiHighlighter
                addDefaultStyles={false}
                as="div"
                colorReplacements={SHIKI_COLOR_REPLACEMENTS}
                defaultColor="light-dark()"
                delay={120}
                language="html"
                showLanguage={false}
                theme={SHIKI_THEME}
              >
                {source}
              </ShikiHighlighter>
            </pre>
          </CodeCardBody>
        ) : (
          <PreviewFrame
            className="block h-[22.5rem] w-full border-0 bg-white"
            reloadKey={reloadKey}
            source={source}
          />
        )}
      </CodeCard>

      <Dialog onOpenChange={setExpanded} open={expanded}>
        <DialogContent className="h-[90vh] w-[92vw] max-w-[92vw] overflow-hidden p-0">
          {/* Title is required for a11y but the frame is the content. */}
          <DialogTitle className="sr-only">{t.preview.tab}</DialogTitle>
          <PreviewFrame className="block size-full border-0 bg-white" reloadKey={reloadKey} source={source} />
        </DialogContent>
      </Dialog>
    </>
  )
}

export const HtmlPreview = memo(HtmlPreviewImpl)
