import type { JSONRPCMessage, Transport } from '@modelcontextprotocol/client'
import { JSONRPCMessageSchema } from '@modelcontextprotocol/core'
import { AppBridge } from '@modelcontextprotocol/ext-apps/app-bridge'
import { type RefObject, useEffect, useLayoutEffect, useRef } from 'react'

import { uiMessageToPrompt } from '@/lib/hosted-artifacts'

/**
 * The MCP Apps host side of one hosted artifact frame, on the official
 * ext-apps `AppBridge`.
 *
 * The frame is a real origin (`allow-same-origin`), so messages are pinned to
 * it both ways: the host posts with that exact `targetOrigin` (a frame that
 * navigated elsewhere never receives the tool result and its ticket), and
 * accepts a message only when it comes from this iframe's window AND from
 * that origin.
 */
export class PinnedOriginTransport implements Transport {
  onclose?: () => void
  onerror?: (error: Error) => void
  onmessage?: (message: JSONRPCMessage) => void
  sessionId?: string

  private readonly listener = (event: MessageEvent) => {
    if (event.source !== this.target || event.origin !== this.origin) {
      return
    }

    const parsed = JSONRPCMessageSchema.safeParse(event.data)

    if (parsed.success) {
      this.onmessage?.(parsed.data as JSONRPCMessage)
    } else {
      this.onerror?.(new Error('Invalid JSON-RPC message from the artifact frame'))
    }
  }

  constructor(
    private readonly target: Window,
    private readonly origin: string,
    private readonly host: Window = window
  ) {}

  async start(): Promise<void> {
    this.host.addEventListener('message', this.listener)
  }

  async send(message: JSONRPCMessage): Promise<void> {
    this.target.postMessage(message, this.origin)
  }

  async close(): Promise<void> {
    this.host.removeEventListener('message', this.listener)
    this.onclose?.()
  }
}

export interface HostedArtifactBridgeOptions {
  /** The tool call's arguments, sent to the view as its tool input. */
  args: Record<string, unknown>
  /** Exact origin of the framed artifact. */
  origin: string
  /** Raw text the tool returned, sent to the view as its tool result. */
  resultText: string
  /** Height the view asks for (`ui/notifications/size-changed`). */
  onHeight?: (height: number) => void
  /** Offer a message the view asks to send for the user to confirm; false when it cannot be
   *  offered. Never sends on its own: without it every `ui/message` is refused. */
  offerMessage?: (text: string) => boolean
  now?: () => number
}

const HOST_INFO = { name: 'Hermes Desktop', version: '1' }

/** Connect an AppBridge to the iframe once it is mounted. */
export function useHostedArtifactBridge(
  frame: RefObject<HTMLIFrameElement | null>,
  options: HostedArtifactBridgeOptions
) {
  const latest = useRef(options)

  useLayoutEffect(() => {
    latest.current = options
  })

  const { origin } = options

  useEffect(() => {
    const target = frame.current?.contentWindow

    if (!target) {
      return
    }

    const bridge = createHostedArtifactBridge(target, origin, () => latest.current)

    // A frame that never speaks the protocol (plain content) is fine: the
    // bridge just stays idle.
    void bridge.ready.catch(() => undefined)

    return () => {
      void bridge.close()
    }
  }, [frame, origin])
}

export interface HostedArtifactBridgeHandle {
  bridge: AppBridge
  close: () => Promise<void>
  ready: Promise<void>
}

/**
 * Wire one AppBridge: answers `ui/initialize`, sends the tool input and result
 * once the view is initialized, offers a `ui/message` to the user (an artifact
 * may come from another user, so it never sends on its own), and follows the
 * view's size requests.
 */
export function createHostedArtifactBridge(
  target: Window,
  origin: string,
  options: () => HostedArtifactBridgeOptions,
  host: Window = window
): HostedArtifactBridgeHandle {
  const bridge = new AppBridge(
    null,
    HOST_INFO,
    { message: { text: {} } },
    { hostContext: { displayMode: 'inline', platform: 'desktop' } }
  )

  let lastAcceptedAt = Number.NEGATIVE_INFINITY

  bridge.onmessage = async params => {
    const current = options()
    const now = (current.now ?? Date.now)()
    const decision = uiMessageToPrompt(params, now, lastAcceptedAt)

    if (decision.kind === 'reject') {
      return { isError: true }
    }

    if (!current.offerMessage?.(decision.text)) {
      return { isError: true }
    }

    lastAcceptedAt = now

    return {}
  }

  bridge.oninitialized = () => {
    const current = options()

    void bridge.sendToolInput({ arguments: current.args })
    void bridge.sendToolResult({ content: [{ text: current.resultText, type: 'text' }] })
  }

  bridge.onsizechange = ({ height }) => {
    if (typeof height === 'number' && Number.isFinite(height) && height > 0) {
      options().onHeight?.(height)
    }
  }

  const transport = new PinnedOriginTransport(target, origin, host)
  const ready = bridge.connect(transport)

  return {
    bridge,
    close: () => bridge.close(),
    ready
  }
}
