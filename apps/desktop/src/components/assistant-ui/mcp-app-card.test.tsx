import { type ToolCallMessagePartProps } from '@assistant-ui/react'
import { cleanup, render, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { $mcpAppUserMessage, clearMcpAppUserMessage, consumeStagedModelContext } from '@/store/mcp-app'

import { hasMcpUi, McpAppCard } from './mcp-app-card'

const { requestGateway } = vi.hoisted(() => ({ requestGateway: vi.fn() }))

vi.mock('@/app/gateway/hooks/use-gateway-request', () => ({
    useGatewayRequest: () => ({ requestGateway })
}))

const uiResult = {
    ui: {
        server: 'utp',
        uri: 'ui://utp/catalog-search',
        mimeType: 'text/html;profile=mcp-app',
        html: '<!DOCTYPE html><html><head><title>t</title></head><body>card</body></html>',
        csp: {
            scriptSrc: "'unsafe-inline' 'unsafe-eval'",
            connectDomains: ['*.alicdn.com'],
            resourceDomains: ['*.alicdn.com']
        }
    }
}

function renderCard(result: unknown) {
    // McpAppCard only reads `result`; the rest of ToolCallMessagePartProps is unused here.
    return render(<McpAppCard {...({ result } as unknown as ToolCallMessagePartProps)} />)
}

afterEach(() => {
    cleanup()
    requestGateway.mockReset()
    clearMcpAppUserMessage()
    consumeStagedModelContext()
})

describe('hasMcpUi', () => {
    it('detects a UI card payload and ignores plain results', () => {
        expect(hasMcpUi(uiResult)).toBe(true)
        expect(hasMcpUi({ result: 'plain' })).toBe(false)
        expect(hasMcpUi(undefined)).toBe(false)
        // resource missing uri/text -> not a card
        expect(hasMcpUi({ ui: { server: 'utp' } })).toBe(false)
    })
})

describe('McpAppCard', () => {
    it('renders a sandboxed iframe with the server CSP injected and the html inlined', () => {
        const { container } = renderCard(uiResult)
        const iframe = container.querySelector('iframe')
        expect(iframe).toBeTruthy()
        expect(iframe!.getAttribute('sandbox')).toContain('allow-scripts')
        const doc = iframe!.getAttribute('srcdoc') || ''
        expect(doc).toContain('http-equiv="Content-Security-Policy"')
        expect(doc).toContain('*.alicdn.com')
        expect(doc).toContain('<body>card</body>')
    })

    it('renders nothing when the result carries no UI card', () => {
        const { container } = renderCard({ result: 'plain' })
        expect(container.querySelector('iframe')).toBeNull()
    })

    it('bridges a JSON-RPC request to mcp.app.request and posts the response back into the iframe', async () => {
        const response = { jsonrpc: '2.0', id: 5, result: { isError: false, content: [], structuredContent: { id: 'p1' } } }
        requestGateway.mockResolvedValue({ response })

        const { container } = renderCard(uiResult)
        const iframe = container.querySelector('iframe') as HTMLIFrameElement
        const postSpy = vi.spyOn(iframe.contentWindow!, 'postMessage')

        const msg = { jsonrpc: '2.0', id: 5, method: 'tools/call', params: { name: 'utp_catalog_product', arguments: { product_id: 'p1' } } }
        window.dispatchEvent(new MessageEvent('message', { data: msg, source: iframe.contentWindow }))

        await waitFor(() =>
            expect(requestGateway).toHaveBeenCalledWith('mcp.app.request', { server: 'utp', message: msg })
        )
        await waitFor(() => expect(postSpy).toHaveBeenCalledWith(response, '*'))
    })

    it('ignores messages that do not originate from its own iframe', async () => {
        renderCard(uiResult)
        window.dispatchEvent(
            new MessageEvent('message', { data: { jsonrpc: '2.0', id: 1, method: 'tools/call' }, source: window })
        )
        await new Promise(resolve => setTimeout(resolve, 20))
        expect(requestGateway).not.toHaveBeenCalled()
    })

    it('answers ui/initialize locally with a host result and does not call the gateway', async () => {
        const { container } = renderCard(uiResult)
        const iframe = container.querySelector('iframe') as HTMLIFrameElement
        const postSpy = vi.spyOn(iframe.contentWindow!, 'postMessage')

        window.dispatchEvent(
            new MessageEvent('message', {
                data: {
                    jsonrpc: '2.0',
                    id: 2,
                    method: 'ui/initialize',
                    params: { protocolVersion: '2026-01-26', appInfo: { name: 'UCP Login', version: '5.0.0' } }
                },
                source: iframe.contentWindow
            })
        )

        await waitFor(() => {
            const initReply = postSpy.mock.calls.find(c => (c[0] as { id?: number })?.id === 2)
            expect(initReply).toBeTruthy()
            const frame = initReply![0] as { result?: { protocolVersion?: string; hostInfo?: unknown } }
            expect(frame.result?.protocolVersion).toBe('2026-01-26')
            expect(frame.result?.hostInfo).toBeTruthy()
        })
        expect(requestGateway).not.toHaveBeenCalled()
    })

    it('hands the tool result to the card as ui/initialize lastToolResult + sessionId', async () => {
        // Referenced-form cards (utp catalog v5) render their initial view from
        // `lastToolResult` instead of falling back to an empty search page.
        const resultWithData = {
            ...uiResult,
            result: 'Found 2 products',
            structuredContent: { body: { products: [{ id: 'p1' }] }, session_id: 'sess-9' }
        }

        const { container } = renderCard(resultWithData)
        const iframe = container.querySelector('iframe') as HTMLIFrameElement
        const postSpy = vi.spyOn(iframe.contentWindow!, 'postMessage')

        window.dispatchEvent(
            new MessageEvent('message', {
                data: { jsonrpc: '2.0', id: 3, method: 'ui/initialize', params: {} },
                source: iframe.contentWindow
            })
        )

        await waitFor(() => {
            const initReply = postSpy.mock.calls.find(c => (c[0] as { id?: number })?.id === 3)
            expect(initReply).toBeTruthy()

            const frame = initReply![0] as {
                result?: { lastToolResult?: { structuredContent?: unknown; content?: unknown }; sessionId?: string }
            }

            expect(frame.result?.lastToolResult?.structuredContent).toEqual(resultWithData.structuredContent)
            expect(frame.result?.lastToolResult?.content).toEqual([{ type: 'text', text: 'Found 2 products' }])
            expect(frame.result?.sessionId).toBe('sess-9')
        })
        expect(requestGateway).not.toHaveBeenCalled()
    })

    it('resizes the iframe on ui/notifications/size-changed without calling the gateway', async () => {
        const { container } = renderCard(uiResult)
        const iframe = container.querySelector('iframe') as HTMLIFrameElement
        window.dispatchEvent(
            new MessageEvent('message', {
                data: { jsonrpc: '2.0', method: 'ui/notifications/size-changed', params: { height: 777 } },
                source: iframe.contentWindow
            })
        )
        await waitFor(() => expect(iframe.style.height).toBe('777px'))
        expect(requestGateway).not.toHaveBeenCalled()
    })

    it('routes ui/update-model-context + ui/message per spec: silent snapshot + visible message (D10/D12)', async () => {
        const { container } = renderCard(uiResult)
        const iframe = container.querySelector('iframe') as HTMLIFrameElement

        // Two context updates from the same view: overwrite, not accumulate.
        for (const text of ['旧快照 checkout_id=ck-0', '[卡片已创建结账单] checkout_id=ck-1']) {
            window.dispatchEvent(
                new MessageEvent('message', {
                    data: {
                        jsonrpc: '2.0',
                        id: 7,
                        method: 'ui/update-model-context',
                        params: { content: [{ type: 'text', text }] }
                    },
                    source: iframe.contentWindow
                })
            )
        }

        window.dispatchEvent(
            new MessageEvent('message', {
                data: {
                    jsonrpc: '2.0',
                    id: 8,
                    method: 'ui/message',
                    params: { role: 'user', content: [{ type: 'text', text: '帮我下单' }] }
                },
                source: iframe.contentWindow
            })
        )

        await waitFor(() => {
            // Visible message carries ONLY the ui/message text…
            expect($mcpAppUserMessage.get()?.text).toBe('帮我下单')
        })
        // …while the model context slot holds the LAST snapshot only.
        expect(consumeStagedModelContext()).toBe('[卡片已创建结账单] checkout_id=ck-1')
        // Host handles these locally; nothing is proxied to the MCP server.
        expect(requestGateway).not.toHaveBeenCalled()
    })
})
