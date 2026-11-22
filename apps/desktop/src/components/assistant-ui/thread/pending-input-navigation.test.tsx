import {
  AssistantRuntimeProvider,
  type ThreadMessage,
  type ToolCallMessagePartProps,
  useExternalStoreRuntime
} from '@assistant-ui/react'
import { useStore } from '@nanostores/react'
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { atom } from 'nanostores'
import { type ReactNode, useRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { ScrollToBottomButton } from '@/app/chat/scroll-to-bottom-button'
import { PRIMARY_SESSION_VIEW, SessionViewProvider } from '@/app/chat/session-view'
import { CatalogInstallTool } from '@/components/assistant-ui/catalog-install-tool'
import { ClarifyToolPending } from '@/components/assistant-ui/clarify/pending'
import { ConnectorOffer } from '@/components/assistant-ui/connector-tool'
import { McpSetupPending } from '@/components/assistant-ui/mcp-setup-tool'
import { SetupBeatTool } from '@/components/assistant-ui/setup-beat/setup-beat-tool'
import { ThemeTool } from '@/components/assistant-ui/theme-picker/theme-tool'
import { PendingApprovalStack } from '@/components/assistant-ui/tool/approval'
import { clearClarifyRequest, sessionClarifyRequest, setClarifyRequest } from '@/store/clarify'
import {
  $connectionRequests,
  clearConnectionRequest,
  normalizeConnectionRequest,
  setConnectionRequest
} from '@/store/connection-request'
import { clearAllPrompts, clearApprovalRequest, setApprovalRequest } from '@/store/prompts'
import { $setupBeatRequests, clearSetupBeat, parkSetupBeat } from '@/store/setup-beat'
import { $themePicks, clearThemePick, parkThemePick } from '@/store/theme-pick'
import { onScrollToBottomRequest, resetThreadScroll, setThreadAtBottom } from '@/store/thread-scroll'
import { stubResizeObserver } from '@/test/jsdom'
import { ThemeProvider } from '@/themes/context'

import { PendingInputNavigationProvider, usePendingInputReveal } from './pending-input-navigation'
import { TranscriptWindowProvider } from './transcript-window'

stubResizeObserver()

const SESSION = 'pending-nav'
const view = { ...PRIMARY_SESSION_VIEW, $runtimeId: atom(SESSION), $storedId: atom(SESSION), $busy: atom(true) }
const QUESTIONS = [{ choices: ['Yes', 'No'], multiSelect: false, qid: 'q1', question: 'Ship it?' }]
const CLARIFY = { requestId: 'clarify-nav', sessionId: SESSION, questions: QUESTIONS }

const OPERATION = {
  deadline_at: 1_800_000_000,
  op_id: 'connect-nav',
  seq: 1,
  timeout_seconds: 600,
  tool_call_id: 'connection-call'
}

function part(toolName: string, toolCallId: string, args: ToolCallMessagePartProps['args']): ToolCallMessagePartProps {
  return {
    addResult: vi.fn(),
    args,
    argsText: JSON.stringify(args),
    isError: false,
    respondToApproval: vi.fn(),
    result: undefined,
    resume: vi.fn(),
    status: { type: 'running' },
    toolCallId,
    toolName,
    type: 'tool-call'
  }
}

function ClarifyCard({ preview = false }: { preview?: boolean }) {
  return (
    <ClarifyToolPending
      fromArgs={{ questions: QUESTIONS }}
      onAnswered={() => {}}
      request={preview ? null : sessionClarifyRequest(SESSION).get()}
      undelivered={false}
    />
  )
}

function restoreConnection(kind: 'connector' | 'mcp' | 'skill') {
  const request = normalizeConnectionRequest(
    {
      ...OPERATION,
      targets: [
        {
          action: kind === 'connector' ? 'connect' : 'install',
          display: 'Inbox',
          kind,
          name: kind === 'connector' ? 'gmail' : 'inbox',
          state: 'pending'
        }
      ]
    },
    SESSION
  )

  if (!request) {
    throw new Error('Invalid connection fixture')
  }

  setConnectionRequest(request)

  return request
}

const noOp = () => {}
const neverBlocked = () => false

function Pane({
  children,
  move,
  historical = false,
  returnToLatest = noOp
}: {
  children: ReactNode
  move: (viewport: HTMLElement, delta: number) => void
  historical?: boolean
  returnToLatest?: () => void
}) {
  const viewport = useRef<HTMLDivElement>(null)
  const content = useRef<HTMLDivElement>(null)
  const runtime = useExternalStoreRuntime<ThreadMessage>({ messages: [], isRunning: false, onNew: async () => {} })

  usePendingInputReveal({
    viewport,
    content,
    visible: true,
    following: true,
    transcriptKey: SESSION,
    ready: true,
    historical,
    returnToLatest,
    blocked: neverBlocked,
    measure: measureOffscreen,
    move
  })

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div ref={viewport}>
        <div ref={content}>{children}</div>
      </div>
      <div aria-label="Transcript navigation" role="group">
        <TranscriptWindowProvider value={{ olderAvailable: false, expandWindow: noOp, isHistorical: historical }}>
          <ScrollToBottomButton sessionId={SESSION} />
        </TranscriptWindowProvider>
      </div>
    </AssistantRuntimeProvider>
  )
}

function measureOffscreen(node: HTMLElement) {
  const marker = node.closest<HTMLElement>('[data-test-offset]')

  return { visible: false, delta: Number(marker?.dataset.testOffset) }
}

function TestPane({
  children,
  move,
  historical,
  returnToLatest
}: {
  children: ReactNode
  move: (viewport: HTMLElement, delta: number) => void
  historical?: boolean
  returnToLatest?: () => void
}) {
  return (
    <ThemeProvider>
      <SessionViewProvider value={view}>
        <PendingInputNavigationProvider active={false} sessionId={SESSION}>
          <Pane historical={historical} move={move} returnToLatest={returnToLatest}>
            {children}
          </Pane>
        </PendingInputNavigationProvider>
      </SessionViewProvider>
    </ThemeProvider>
  )
}

const approval = {
  park: () =>
    setApprovalRequest({
      command: 'pwd',
      description: 'Read working directory',
      requestId: 'approve-nav',
      sessionId: SESSION
    }),
  clear: () => clearApprovalRequest(SESSION, 'approve-nav'),
  card: () => <PendingApprovalStack />,
  label: 'Approval needed'
}

const clarify = {
  park: () => setClarifyRequest(CLARIFY),
  clear: () => clearClarifyRequest(CLARIFY.requestId, SESSION),
  card: () => <ClarifyCard />,
  label: 'Needs your input'
}

function LiveConnector() {
  const request = useStore($connectionRequests)[SESSION]

  return request ? <ConnectorOffer owner={{ connectionId: null, profile: 'default' }} request={request} /> : null
}

function LiveTheme() {
  const request = useStore($themePicks)[SESSION]

  return request ? <ThemeTool {...part('set_theme', 'theme-call', { picker: 'accent' })} /> : null
}

const connector = {
  park: () => restoreConnection('connector'),
  clear: () => clearConnectionRequest(OPERATION.op_id, SESSION),
  card: () => <LiveConnector />,
  label: 'Needs your input'
}

const setup = {
  park: () =>
    parkSetupBeat({
      beat: 'name',
      card: {},
      question: 'What should I call you?',
      requestId: 'setup-nav',
      sessionId: SESSION,
      toolCallId: 'setup-call'
    }),
  clear: () => clearSetupBeat('setup-nav', SESSION),
  card: () => <SetupBeatTool {...part('setup_beat', 'setup-call', {})} />,
  label: 'Needs your input'
}

const theme = {
  park: () => parkThemePick({ picker: 'accent', requestId: 'theme-nav', sessionId: SESSION }),
  clear: () => clearThemePick('theme-nav', SESSION),
  card: () => <LiveTheme />,
  label: 'Needs your input'
}

const cases = {
  approval,
  clarify,
  connector,
  MCP: {
    ...connector,
    park: () => restoreConnection('mcp'),
    card: () => <McpSetupPending {...part('manage_connections', OPERATION.tool_call_id, { action: 'install' })} />
  },
  catalog: {
    ...connector,
    park: () => restoreConnection('skill'),
    card: () => <CatalogInstallTool {...part('manage_catalog', OPERATION.tool_call_id, { action: 'install' })} />
  },
  setup,
  theme
}

afterEach(() => {
  cleanup()
  resetThreadScroll(SESSION)
  clearAllPrompts()
  clearClarifyRequest()
  clearConnectionRequest()
  $setupBeatRequests.set({})
  $themePicks.set({})
})

describe('pending question navigation', () => {
  it.each(Object.entries(cases))(
    'jumps to the live %s card and drops the pill when its request clears',
    async (_, fixture) => {
      fixture.park()
      const move = vi.fn()
      render(
        <TestPane move={move}>
          <div data-test-offset="125">{fixture.card()}</div>
        </TestPane>
      )

      fireEvent.click(
        within(screen.getByRole('group', { name: 'Transcript navigation' })).getByRole('button', {
          name: fixture.label
        })
      )
      await waitFor(() => expect(move).toHaveBeenCalledWith(expect.any(HTMLElement), 125))

      act(() => void fixture.clear())
      expect(
        within(screen.getByRole('group', { name: 'Transcript navigation' })).queryByRole('button', {
          name: fixture.label
        })
      ).toBeNull()
    }
  )

  it('jumps in priority order as each higher-priority request settles', async () => {
    const ordered = [approval, clarify, connector, setup, theme]
    ordered.forEach(fixture => fixture.park())
    const move = vi.fn()

    render(
      <TestPane move={move}>
        {ordered.map((fixture, index) => (
          <div data-test-offset={100 + index} key={index}>
            {fixture.card()}
          </div>
        ))}
      </TestPane>
    )

    for (const [index, fixture] of ordered.entries()) {
      fireEvent.click(
        within(screen.getByRole('group', { name: 'Transcript navigation' })).getByRole('button', {
          name: fixture.label
        })
      )
      await waitFor(() => expect(move).toHaveBeenLastCalledWith(expect.any(HTMLElement), 100 + index))
      act(() => void fixture.clear())
    }
  })

  it('never jumps to a disabled clarify preview, even while a real request is parked', async () => {
    clarify.park()
    const move = vi.fn()
    render(
      <TestPane move={move}>
        <ClarifyCard preview />
      </TestPane>
    )

    expect(screen.queryByRole('button', { name: clarify.label })).toBeNull()
    const bottom = vi.fn()
    const stop = onScrollToBottomRequest(bottom, SESSION)

    try {
      act(() => setThreadAtBottom(false, SESSION))
      fireEvent.click(screen.getByRole('button', { name: clarify.label }))
      await act(() => new Promise(resolve => setTimeout(resolve, 130)))

      expect(bottom).toHaveBeenCalledOnce()
      expect(move).not.toHaveBeenCalled()
    } finally {
      stop()
    }
  })

  it('skips a timed-out connection receipt and uses the next live card', async () => {
    const request = restoreConnection('connector')
    setConnectionRequest({ ...request, settled: true, settledBy: 'deadline' })
    setup.park()
    const move = vi.fn()

    render(
      <TestPane move={move}>
        <div data-test-offset="111">
          <LiveConnector />
        </div>
        <div data-test-offset="222">{setup.card()}</div>
      </TestPane>
    )

    fireEvent.click(screen.getByRole('button', { name: setup.label }))
    await waitFor(() => expect(move).toHaveBeenCalledWith(expect.any(HTMLElement), 222))
    expect(move).not.toHaveBeenCalledWith(expect.any(HTMLElement), 111)
  })

  it('does not jump to a settled picker row during a request replay', async () => {
    theme.park()
    const move = vi.fn()
    render(
      <TestPane move={move}>
        <ThemeTool
          {...part('set_theme', 'theme-call', { picker: 'accent' })}
          result={{ cancelled: true, picked: { accent: 'violet' } }}
        />
      </TestPane>
    )

    expect(screen.queryByRole('button', { name: theme.label })).toBeNull()
    act(() => setThreadAtBottom(false, SESSION))
    fireEvent.click(screen.getByRole('button', { name: theme.label }))
    await act(() => new Promise(resolve => setTimeout(resolve, 130)))

    expect(move).not.toHaveBeenCalled()
  })

  it('uses only the clicked pane’s card even when both panes show the same runtime', async () => {
    clarify.park()
    const first = vi.fn()
    const second = vi.fn()
    render(
      <>
        <section aria-label="First pane">
          <TestPane move={first}>
            <div data-test-offset="111">{clarify.card()}</div>
          </TestPane>
        </section>
        <section aria-label="Second pane">
          <TestPane move={second}>
            <div data-test-offset="222">{clarify.card()}</div>
          </TestPane>
        </section>
      </>
    )

    fireEvent.click(
      within(screen.getByRole('region', { name: 'Second pane' })).getByRole('button', { name: clarify.label })
    )
    await waitFor(() => expect(second).toHaveBeenCalledWith(expect.any(HTMLElement), 222))
    expect(first).not.toHaveBeenCalled()
  })

  it('returns from a historical page before handling the explicit jump', async () => {
    clarify.park()
    const latest = vi.fn()
    const move = vi.fn()

    const { rerender } = render(
      <TestPane historical move={move} returnToLatest={latest}>
        {null}
      </TestPane>
    )

    fireEvent.click(screen.getByRole('button', { name: clarify.label }))
    expect(latest).toHaveBeenCalledOnce()
    expect(move).not.toHaveBeenCalled()

    rerender(
      <TestPane move={move}>
        <div data-test-offset="125">{clarify.card()}</div>
      </TestPane>
    )
    await waitFor(() => expect(move).toHaveBeenCalledWith(expect.any(HTMLElement), 125))
  })
})
