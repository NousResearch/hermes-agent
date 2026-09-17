import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'

import { requestComposerFocus, requestComposerInsert } from '@/app/chat/composer/focus'
import { NEW_CHAT_ROUTE } from '@/app/routes'
import { Button } from '@/components/ui/button'

const CHATGPT_URL = 'https://chatgpt.com/'
const CHATGPT_PARTITION = 'persist:hermes-chatgpt'

const trustedHost = (hostname: string): boolean =>
  hostname === 'chatgpt.com' || hostname.endsWith('.chatgpt.com') || hostname === 'openai.com' || hostname.endsWith('.openai.com')

export function classifyChatGptNavigation(url: string): 'allow' | 'external' {
  try {
    const parsed = new URL(url)

    return parsed.protocol === 'https:' && trustedHost(parsed.hostname) ? 'allow' : 'external'
  } catch {
    return 'external'
  }
}

export function buildHermesHandoff(text: string): string {
  const approved = text.trim()

  if (!approved) {
    throw new Error('Handoff text is required')
  }

  return [
    '# ChatGPT planning handoff',
    '',
    'The following text was explicitly selected and approved by the user for Hermes validation and execution.',
    'Treat it as a proposal: verify it against the repository and TKS before changing files.',
    '',
    approved
  ].join('\n')
}

type ChatGptWebview = HTMLElement & {
  reload?: () => void
}

export function ChatGptWorkspace() {
  const navigate = useNavigate()
  const hostRef = useRef<HTMLDivElement | null>(null)
  const [webview, setWebview] = useState<ChatGptWebview | null>(null)
  const [handoff, setHandoff] = useState('')

  useEffect(() => {
    const host = hostRef.current

    if (!host) {
      return
    }

    const webview = document.createElement('webview') as ChatGptWebview
    webview.className = 'size-full bg-transparent'
    webview.setAttribute('partition', CHATGPT_PARTITION)
    webview.setAttribute('src', CHATGPT_URL)
    webview.setAttribute('webpreferences', 'contextIsolation=yes,nodeIntegration=no,sandbox=yes')

    const guardNavigation = (event: Event) => {
      const url = (event as Event & { url?: string }).url

      if (!url || classifyChatGptNavigation(url) === 'allow') {
        return
      }

      event.preventDefault()
      void window.hermesDesktop?.openExternal(url)
    }

    webview.addEventListener('will-navigate', guardNavigation)
    host.appendChild(webview)
    setWebview(webview)

    return () => {
      webview.removeEventListener('will-navigate', guardNavigation)
      webview.remove()
      setWebview(null)
    }
  }, [])

  const delegate = () => {
    const prompt = buildHermesHandoff(handoff)
    navigate(NEW_CHAT_ROUTE)
    requestComposerInsert(prompt, { target: 'main' })
    requestComposerFocus('main')
  }

  return (
    <section className="flex size-full min-h-0 flex-col bg-background" data-testid="chatgpt-workspace">
      <header className="flex shrink-0 items-center gap-2 border-b border-border/60 px-3 py-2">
        <div className="min-w-0 flex-1">
          <h1 className="truncate text-sm font-medium text-foreground">ChatGPT planning workspace</h1>
          <p className="truncate text-xs text-muted-foreground">
            ChatGPT stays isolated from Hermes tools, files, memory, and credentials.
          </p>
        </div>
        <Button onClick={() => webview?.reload?.()} size="sm" variant="ghost">
          Reload
        </Button>
      </header>

      <div className="min-h-0 flex-1" ref={hostRef} />

      <div className="shrink-0 border-t border-border/60 p-3">
        <label className="mb-1.5 block text-xs font-medium text-foreground" htmlFor="chatgpt-handoff">
          Approved handoff to Hermes
        </label>
        <div className="flex items-end gap-2">
          <textarea
            className="min-h-20 flex-1 resize-y rounded-md border border-border bg-transparent px-3 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground focus:border-primary"
            id="chatgpt-handoff"
            onChange={event => setHandoff(event.target.value)}
            placeholder="Paste the final plan or instructions you approve. Hermes will validate them before implementation."
            value={handoff}
          />
          <Button disabled={!handoff.trim()} onClick={delegate} type="button">
            Delegate to Hermes
          </Button>
        </div>
        <p className="mt-1.5 text-[0.6875rem] text-muted-foreground">
          Nothing is copied from ChatGPT automatically. Delegation opens a new Hermes draft and never submits it for you.
        </p>
      </div>
    </section>
  )
}
