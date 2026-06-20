import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { setWorkflowCopilotExpanded, setWorkflowCopilotOpen } from '@/store/workflow'

import { WorkflowView } from './index'

function setDesktopBridge(value: unknown) {
  Object.defineProperty(window, 'hermesDesktop', {
    configurable: true,
    value
  })
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  setWorkflowCopilotExpanded(false)
  setWorkflowCopilotOpen(false)
  Reflect.deleteProperty(window, 'hermesDesktop')
})

describe('WorkflowView backend startup', () => {
  it('does not mount the Langflow webview when the desktop workflow bridge is unavailable', async () => {
    setDesktopBridge({})

    const rendered = render(<WorkflowView />)

    await waitFor(() => {
      expect(screen.getByRole('alert')).toBeTruthy()
    })

    expect(rendered.container.querySelector('webview')).toBeNull()
  })

  it('starts Langflow even when the EasyHermes account is not logged in', async () => {
    const start = vi.fn(async () => ({
      error: null,
      external: false,
      pid: 1234,
      root: '/Volumes/D/kari-all/langflow',
      state: 'ready',
      url: 'http://127.0.0.1:7860'
    }))
    setDesktopBridge({
      workflow: {
        authStatus: vi.fn(async () => ({
          cloudBaseUrl: 'http://127.0.0.1:8900',
          cloudReachable: false,
          error: 'Kari hub is not reachable: http://127.0.0.1:8900. Please log in to Workflow again or start the local hub.',
          loggedIn: false
        })),
        start,
        status: vi.fn()
      }
    })

    const rendered = render(<WorkflowView />)

    await waitFor(() => expect(start).toHaveBeenCalled())

    const srcs = [...rendered.container.querySelectorAll('webview')].map(node => node.getAttribute('src'))
    expect(srcs).toContain('http://127.0.0.1:7860')
    expect(screen.queryByText('登录工作流')).toBeNull()
  })

  it('keeps the knowledge view mounted when webview script injection runs before dom-ready', async () => {
    setDesktopBridge({
      workflow: {
        authStatus: vi.fn(async () => ({
          cloudBaseUrl: 'http://127.0.0.1:8900',
          cloudReachable: true,
          error: null,
          loggedIn: true
        })),
        start: vi.fn(async () => ({
          error: null,
          external: false,
          pid: 1234,
          root: '/Volumes/D/kari-all/langflow',
          state: 'ready',
          url: 'http://127.0.0.1:7860'
        })),
        status: vi.fn()
      }
    })

    const rendered = render(<WorkflowView view="knowledge" />)

    await waitFor(() => {
      expect(rendered.container.querySelector('webview')).toBeTruthy()
    })

    const webview = rendered.container.querySelector('webview') as HTMLElement & {
      executeJavaScript: () => Promise<unknown>
    }
    webview.executeJavaScript = vi.fn(() => {
      throw new Error('The WebView must be attached to the DOM and the dom-ready event emitted before this method can be called.')
    })

    expect(() => {
      fireEvent(webview, new Event('did-stop-loading'))
    }).not.toThrow()
    expect(rendered.container.querySelector('webview')).toBe(webview)
  })

  it('keeps backend error recovery inside EasyHermes instead of offering an external browser action', async () => {
    setDesktopBridge({
      workflow: {
        start: vi.fn(async () => ({
          error: 'Langflow did not become reachable',
          external: false,
          pid: null,
          root: '/Volumes/D/kari-all/langflow',
          state: 'error',
          url: 'http://127.0.0.1:7860'
        })),
        status: vi.fn()
      }
    })

    render(<WorkflowView />)

    await waitFor(() => {
      expect(screen.getByText('Langflow did not become reachable')).toBeTruthy()
    })

    expect(screen.queryByText('Open in browser')).toBeNull()
  })

  it('opens the right-side copilot as the Hermes /copilot bubble page, not the xterm /chat page', async () => {
    setDesktopBridge({
      getConnection: vi.fn(async () => ({
        baseUrl: 'http://127.0.0.1:9119'
      })),
      workflow: {
        start: vi.fn(async () => ({
          error: null,
          external: false,
          pid: 1234,
          root: '/Volumes/D/kari-all/langflow',
          state: 'ready',
          url: 'http://127.0.0.1:7860'
        })),
        status: vi.fn()
      }
    })

    setWorkflowCopilotOpen(true)

    const rendered = render(<WorkflowView />)

    await waitFor(() => {
      const srcs = [...rendered.container.querySelectorAll('webview')].map(node => node.getAttribute('src'))

      expect(srcs).toContain('http://127.0.0.1:9119/copilot')
      expect(srcs).not.toContain('http://127.0.0.1:9119/chat')
    })
  })

  it('does not render a floating copilot toggle over the workflow canvas', async () => {
    setDesktopBridge({
      getConnection: vi.fn(async () => ({
        baseUrl: 'http://127.0.0.1:9119'
      })),
      workflow: {
        start: vi.fn(async () => ({
          error: null,
          external: false,
          pid: 1234,
          root: '/Volumes/D/kari-all/langflow',
          state: 'ready',
          url: 'http://127.0.0.1:7860'
        })),
        status: vi.fn()
      }
    })

    render(<WorkflowView />)

    await waitFor(() => {
      expect(window.hermesDesktop?.getConnection).toHaveBeenCalled()
    })

    expect(screen.queryByRole('button', { name: '爱马仕 Copilot' })).toBeNull()
  })

  it('can expand the right-side copilot while keeping the canvas in the layout flow', async () => {
    setDesktopBridge({
      getConnection: vi.fn(async () => ({
        baseUrl: 'http://127.0.0.1:9119'
      })),
      workflow: {
        start: vi.fn(async () => ({
          error: null,
          external: false,
          pid: 1234,
          root: '/Volumes/D/kari-all/langflow',
          state: 'ready',
          url: 'http://127.0.0.1:7860'
        })),
        status: vi.fn()
      }
    })

    setWorkflowCopilotOpen(true)

    const rendered = render(<WorkflowView />)

    const panel = await screen.findByTestId('workflow-copilot-panel')
    expect(panel.getAttribute('data-expanded')).toBe('false')
    expect(panel.getAttribute('style')).toContain('--workflow-copilot-width: clamp(24rem, 32vw, 30rem)')

    fireEvent.click(screen.getByRole('button', { name: '展开爱马仕 Copilot' }))

    expect(panel.getAttribute('data-expanded')).toBe('true')
    expect(panel.getAttribute('style')).toContain('--workflow-copilot-width: min(46rem, max(28rem, 42vw))')

    const host = await screen.findByTestId('workflow-copilot-host')
    expect(host.className).toContain('w-full')
    expect(host.className).not.toContain('w-[420px]')

    const srcs = [...rendered.container.querySelectorAll('webview')].map(node => node.getAttribute('src'))
    expect(srcs).toContain('http://127.0.0.1:9119/copilot')
  })
})
