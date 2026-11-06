import type { AccountUsageResponse } from '@/types/account-usage'

export interface UsageFrame {
  id: number | string
  method: string
  params: Record<string, unknown>
}

export interface UsageRequest {
  frame: UsageFrame
  peer: AccountUsagePeer
  receivedAt: number
}

/** The external JSON-RPC peer; routing, negotiation and request cancellation stay real. */
export class AccountUsagePeer extends EventTarget {
  static readonly CONNECTING = 0
  static readonly OPEN = 1
  static readonly CLOSING = 2
  static readonly CLOSED = 3
  static requests: UsageRequest[] = []
  static instances: AccountUsagePeer[] = []
  static onUsage: (request: UsageRequest) => void = () => {}
  readyState = AccountUsagePeer.CONNECTING

  constructor(readonly url: string) {
    super()
    AccountUsagePeer.instances.push(this)
    queueMicrotask(() => {
      this.readyState = AccountUsagePeer.OPEN
      this.dispatchEvent(new Event('open'))
      this.receive({
        jsonrpc: '2.0',
        method: 'event',
        params: { type: 'gateway.ready', payload: { session_authority: true } }
      })
    })
  }

  send(data: string): void {
    const frame = JSON.parse(data) as UsageFrame

    if (frame.method === 'session.account_usage') {
      const request = { frame, peer: this, receivedAt: Date.now() }

      AccountUsagePeer.requests.push(request)
      AccountUsagePeer.onUsage(request)

      return
    }

    this.receive({ jsonrpc: '2.0', id: frame.id, result: { value: 'manual', pong: true } })
  }

  answer(frame: UsageFrame, result: AccountUsageResponse): void {
    this.receive({ jsonrpc: '2.0', id: frame.id, result })
  }

  reject(frame: UsageFrame, code: number): void {
    this.receive({ jsonrpc: '2.0', id: frame.id, error: { code, message: 'Fixture RPC failure' } })
  }

  close(): void {
    if (this.readyState === AccountUsagePeer.CLOSED) {
      return
    }

    this.readyState = AccountUsagePeer.CLOSED
    this.dispatchEvent(new CloseEvent('close', { code: 1000 }))
  }

  private receive(frame: unknown): void {
    this.dispatchEvent(new MessageEvent('message', { data: JSON.stringify(frame) }))
  }
}
