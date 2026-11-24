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

// Consecutive transient retries before giving up and waiting for a structural
// resetKey change (the pre-retry behavior). The race heals on the next
// consistent store snapshot, so one retry almost always recovers; the cap
// only bounds a pathological loop where the lookup stays out of bounds.
const MAX_TRANSIENT_RETRIES = 5

interface Props {
  // Changes whenever the message list mutates STRUCTURALLY (ids/roles/count);
  // remounting clears the caught error so the next consistent render recovers
  // silently. Deliberately NOT the per-token signature: this prop reaches
  // every turn's boundary, so a value that ticks with content length would
  // re-render every boundary — and reconcile every turn's subtree — on every
  // streamed token (measured: 540 wasted Block renders per explain() sample
  // with two threads streaming).
  resetKey: string
  children: ReactNode
}

export class MessageRenderBoundary extends Component<Props, { error: Error | null }> {
  state: { error: Error | null } = { error: null }

  private retryTimer: number | null = null

  private transientRetries = 0

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error) {
    // The resetKey path below only recovers on a STRUCTURAL change, but this
    // race also fires mid-turn while ids/roles/count are stable: without a
    // self-retry the boundary renders null for the rest of the turn (or
    // until an unrelated message add/remove). Retry on a timer, not rAF —
    // a parked renderer never fires frames, and a timer always runs.
    if (!isTransientLookupError(error) || this.transientRetries >= MAX_TRANSIENT_RETRIES) {
      return
    }

    if (typeof window === 'undefined') {
      return
    }

    this.transientRetries += 1
    this.retryTimer = window.setTimeout(() => {
      this.retryTimer = null
      this.setState({ error: null })
    }, 0)
  }

  componentDidUpdate(prev: Props, prevState: { error: Error | null }) {
    if (this.state.error && prev.resetKey !== this.props.resetKey) {
      this.setState({ error: null })

      return
    }

    if (prevState.error && !this.state.error) {
      // Recovered (retry or structural reset): reset the retry budget and
      // drop any retry timer the structural reset just made redundant.
      this.transientRetries = 0

      if (this.retryTimer !== null) {
        window.clearTimeout(this.retryTimer)
        this.retryTimer = null
      }
    }
  }

  componentWillUnmount() {
    if (this.retryTimer !== null) {
      window.clearTimeout(this.retryTimer)
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
