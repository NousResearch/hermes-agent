import { Component, type ReactNode } from 'react'

import { useI18n } from '@/i18n'

// `@assistant-ui/store`'s index-keyed child-scope lookup (`tapClientLookup`)
// throws — rather than returning undefined — when a subscriber reads an index
// that the message/parts list no longer has. This races during high-frequency
// store replacement (session switch mid-stream, gateway reconnect replay): a
// subscriber from the previous, longer list is still in React's notification
// queue and reads one slot past the new, shorter array before it can unmount.
// The throw is transient and self-heals on the next consistent snapshot, so it
// is swallowed silently (no fallback flash, no log spam).
// Upstream-tracked: assistant-ui/assistant-ui#4051, #3652.
const isTransientLookupError = (error: unknown): boolean =>
  error instanceof Error && /tapClient(Lookup|Resource).*out of bounds/.test(error.message)

function MessageRenderFallback() {
  const { t } = useI18n()

  return (
    <p
      className="rounded-md border border-border/65 bg-(--composer-fill) px-3 py-2 text-xs text-muted-foreground"
      role="alert"
    >
      {t.assistant.thread.messageRenderFailed}
    </p>
  )
}

interface Props {
  // Changes whenever the message list mutates; remounting clears the caught
  // error so the next consistent render recovers silently.
  resetKey: string
  children: ReactNode
}

export class MessageRenderBoundary extends Component<Props, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    // One malformed message must not unwind to the root boundary and blank
    // the whole app (sidebar, composer, every other session). Contain it here
    // and keep the cause in the console pipeline that main persists to
    // desktop.log so real render bugs still surface.
    if (!isTransientLookupError(error)) {
      console.error('[message-render-boundary]', error)
    }
  }

  componentDidUpdate(prev: Props) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) {
      this.setState({ error: null })
    }
  }

  render() {
    if (this.state.error) {
      return isTransientLookupError(this.state.error) ? null : <MessageRenderFallback />
    }

    return this.props.children
  }
}
