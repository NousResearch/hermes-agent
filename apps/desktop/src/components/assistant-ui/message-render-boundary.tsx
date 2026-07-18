import { Component, type ReactNode } from 'react'

// `@assistant-ui/store`'s index-keyed child-scope lookup (`useClientLookup`)
// throws — rather than returning undefined — when a subscriber reads an index
// that the message/parts list no longer has. This races during high-frequency
// store replacement (session switch mid-stream, gateway reconnect replay): a
// subscriber from the previous, longer list is still in React's notification
// queue and reads one slot past the new, shorter array before it can unmount.
// The throw is transient and self-heals on the next consistent snapshot, but
// without a local boundary it unwinds to the root and blanks the whole app.
// Upstream-tracked: assistant-ui/assistant-ui#4051, #3652.
const isTransientLookupError = (error: unknown): boolean =>
  error instanceof Error && /(useClientLookup|tapClient(Lookup|Resource)).*out of bounds/.test(error.message)

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

  componentDidUpdate(prev: Props) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) {
      this.setState({ error: null })
    }
  }

  render() {
    if (this.state.error) {
      // Only swallow the transient store race; re-throw anything else so real
      // bugs still reach the root error boundary.
      if (!isTransientLookupError(this.state.error)) {
        throw this.state.error
      }

      return null
    }

    return this.props.children
  }
}
