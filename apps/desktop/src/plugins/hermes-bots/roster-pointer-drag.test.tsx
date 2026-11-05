import type * as HermesSdk from '@hermes/plugin-sdk'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { BotRow, GroupRow } from './bot-row'
import { translateBotsIn } from './i18n-test-helper'
import { $draggingBot } from './user-sections'
import { findPointerDropZone, SectionDropZone, useEscapeCancelsBotDrag } from './user-sections-ui'

const { ensureAgent, ensureBotMetadata, notifyError, openRosterBot, requestProfile, warmAgent, warmProfile } =
  vi.hoisted(() => ({
    ensureAgent: vi.fn(),
    ensureBotMetadata: vi.fn(),
    notifyError: vi.fn(),
    openRosterBot: vi.fn(),
    requestProfile: vi.fn(),
    warmAgent: vi.fn(),
    warmProfile: vi.fn()
  }))

vi.mock('@hermes/plugin-sdk', async importOriginal => {
  const sdk = await importOriginal<typeof HermesSdk>()

  return {
    ...sdk,
    host: { ...sdk.host, ensureAgent, notifyError, requestProfile, warmAgent, warmProfile },
    usePluginI18n: () => translateBotsIn('en')
  }
})

vi.mock('./canonical-chat', () => ({
  ensureBotMetadata,
  notifyBotOpenFailure: vi.fn(),
  openBotCanonicalChat: vi.fn(),
  prepareBotSource: vi.fn(),
  PROFILE_SESSION_LIST_LIMIT: 200
}))

vi.mock('./roster-actions', () => ({ openRosterBot }))

beforeAll(() => {
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
  Element.prototype.setPointerCapture = vi.fn()
  Object.defineProperty(window.document, 'elementFromPoint', { configurable: true, value: vi.fn() })
})

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(window.document.elementFromPoint).mockReturnValue(null)
  $draggingBot.set(null)
  openRosterBot.mockResolvedValue(true)
})

afterEach(() => {
  cleanup()
  $draggingBot.set(null)
})

function renderBotRow() {
  render(<BotRow bot={{ connectionId: 'local', name: 'alpha' }} onDelete={vi.fn()} onEdit={vi.fn()} onGroup={vi.fn()} onNewSection={vi.fn()} />)

  return screen.getByRole('button')
}

function beginDrag(row: HTMLElement, pointerId = 1) {
  fireEvent.pointerDown(row, { button: 0, clientX: 0, clientY: 0, isPrimary: true, pointerId })
  fireEvent.pointerMove(row, { clientX: 20, clientY: 20, pointerId })
  expect($draggingBot.get()).toBe('local::alpha')
}

function BotAndZone({ isSource = false, onDropBot }: { isSource?: boolean; onDropBot: (key: string) => void }) {
  useEscapeCancelsBotDrag()

  return (
    <>
      <BotRow
        bot={{ connectionId: 'local', name: 'alpha' }}
        onDelete={vi.fn()}
        onEdit={vi.fn()}
        onGroup={vi.fn()}
        onNewSection={vi.fn()}
      />
      <SectionDropZone isSource={isSource} onDropBot={onDropBot}>
        <div>Target section</div>
      </SectionDropZone>
    </>
  )
}

function renderBotAndZone(onDropBot: (key: string) => void, isSource = false) {
  render(<BotAndZone isSource={isSource} onDropBot={onDropBot} />)
  const row = screen.getByRole('button')
  const zone = screen.getByText('Target section').closest('[data-slot="bots-section"]') as HTMLElement
  vi.mocked(window.document.elementFromPoint).mockReturnValue(zone)

  return { row, zone }
}

describe('roster pointer drag cleanup', () => {
  it('clears the drag when pointer capture is lost', () => {
    const row = renderBotRow()
    beginDrag(row)

    fireEvent.lostPointerCapture(row, { pointerId: 1 })

    expect($draggingBot.get()).toBeNull()
  })

  it('ignores cancellation from a different pointer', () => {
    const row = renderBotRow()
    beginDrag(row)

    fireEvent.pointerCancel(row, { pointerId: 2 })

    expect($draggingBot.get()).toBe('local::alpha')
  })

  it('does not clear another roster row\'s drag', () => {
    const row = renderBotRow()
    beginDrag(row)
    act(() => $draggingBot.set('group::other'))

    fireEvent.pointerCancel(row, { pointerId: 1 })

    expect($draggingBot.get()).toBe('group::other')
  })

  it('clears the drag when the window loses focus', () => {
    const row = renderBotRow()
    beginDrag(row)

    fireEvent.blur(window)

    expect($draggingBot.get()).toBeNull()
  })

  it('clears the drag when the source row unmounts', () => {
    const row = renderBotRow()
    beginDrag(row)

    cleanup()

    expect($draggingBot.get()).toBeNull()
  })

  it('allows a keyboard click after pointer cancellation', () => {
    const row = renderBotRow()
    beginDrag(row)
    fireEvent.pointerCancel(row, { pointerId: 1 })

    fireEvent.click(row, { detail: 0 })

    expect(openRosterBot).toHaveBeenCalledOnce()
  })

  it('keeps a new row\'s drag when the old row cancels', () => {
    const onDropBot = vi.fn()
    render(
      <>
        {(['alpha', 'bravo'] as const).map(name => (
          <BotRow
            bot={{ connectionId: 'local', name }}
            key={name}
            onDelete={vi.fn()}
            onEdit={vi.fn()}
            onGroup={vi.fn()}
            onNewSection={vi.fn()}
          />
        ))}
        <SectionDropZone isSource={false} onDropBot={onDropBot}>
          <div>Target section</div>
        </SectionDropZone>
      </>
    )
    const [alpha, bravo] = screen.getAllByRole('button')
    const zone = screen.getByText('Target section').closest('[data-slot="bots-section"]') as HTMLElement
    vi.mocked(window.document.elementFromPoint).mockReturnValue(zone)
    beginDrag(alpha)
    fireEvent.pointerMove(alpha, { clientX: 30, clientY: 30, pointerId: 1 })
    expect(zone.getAttribute('data-drop-over')).toBe('true')
    fireEvent.pointerDown(bravo, { button: 0, clientX: 0, clientY: 0, isPrimary: true, pointerId: 2 })
    fireEvent.pointerMove(bravo, { clientX: 20, clientY: 20, pointerId: 2 })
    expect($draggingBot.get()).toBe('local::bravo')

    fireEvent.pointerCancel(alpha, { pointerId: 1 })

    expect($draggingBot.get()).toBe('local::bravo')
    expect(zone.getAttribute('data-drop-over')).toBeNull()
  })
})

describe('roster pointer drop contract', () => {
  it('files a dragged bot without opening its chat', () => {
    const onDropBot = vi.fn()
    const { row, zone } = renderBotAndZone(onDropBot)
    beginDrag(row)

    fireEvent.pointerMove(row, { clientX: 30, clientY: 30, pointerId: 1 })
    expect(zone.getAttribute('data-drop-over')).toBe('true')
    fireEvent.pointerUp(row, { clientX: 30, clientY: 30, pointerId: 1 })
    fireEvent.lostPointerCapture(row, { pointerId: 1 })
    fireEvent.click(row)

    expect(onDropBot).toHaveBeenCalledWith('local::alpha')
    expect($draggingBot.get()).toBeNull()
    expect(openRosterBot).not.toHaveBeenCalled()
  })

  it('keeps a sub-threshold gesture as a click', () => {
    const row = renderBotRow()

    fireEvent.pointerDown(row, { button: 0, clientX: 0, clientY: 0, isPrimary: true, pointerId: 1 })
    fireEvent.pointerMove(row, { clientX: 3, clientY: 4, pointerId: 1 })
    fireEvent.pointerUp(row, { clientX: 3, clientY: 4, pointerId: 1 })
    fireEvent.click(row)

    expect($draggingBot.get()).toBeNull()
    expect(openRosterBot).toHaveBeenCalledOnce()
  })

  it('suppresses the synthetic click after an invalid drop', () => {
    const onDropBot = vi.fn()
    const { row } = renderBotAndZone(onDropBot)
    vi.mocked(window.document.elementFromPoint).mockReturnValue(null)
    beginDrag(row)

    fireEvent.pointerUp(row, { clientX: 30, clientY: 30, pointerId: 1 })
    fireEvent.click(row)

    expect(onDropBot).not.toHaveBeenCalled()
    expect(openRosterBot).not.toHaveBeenCalled()
    expect($draggingBot.get()).toBeNull()
  })

  it('refuses a drop after Escape cancels the gesture', () => {
    const onDropBot = vi.fn()
    const { row } = renderBotAndZone(onDropBot)
    beginDrag(row)

    fireEvent.keyDown(window, { key: 'Escape' })
    fireEvent.pointerUp(row, { clientX: 30, clientY: 30, pointerId: 1 })
    fireEvent.click(row)

    expect(onDropBot).not.toHaveBeenCalled()
    expect(openRosterBot).not.toHaveBeenCalled()
    expect($draggingBot.get()).toBeNull()
  })

  it('does not light or drop onto the source section', () => {
    const onDropBot = vi.fn()
    const { row, zone } = renderBotAndZone(onDropBot, true)
    beginDrag(row)

    fireEvent.pointerMove(row, { clientX: 30, clientY: 30, pointerId: 1 })
    expect(zone.getAttribute('data-drop-over')).toBeNull()
    fireEvent.pointerUp(row, { clientX: 30, clientY: 30, pointerId: 1 })

    expect(onDropBot).not.toHaveBeenCalled()
    expect($draggingBot.get()).toBeNull()
  })

  it('files a group through the same pointer gesture', () => {
    const onDropBot = vi.fn()
    const onOpen = vi.fn()
    render(
      <>
        <GroupRow
          active={false}
          group="Room"
          members={[]}
          needsYou={false}
          onDisband={vi.fn()}
          onNewSection={vi.fn()}
          onOpen={onOpen}
        />
        <SectionDropZone isSource={false} onDropBot={onDropBot}>
          <div>Target section</div>
        </SectionDropZone>
      </>
    )
    const row = screen.getByRole('button')
    const zone = screen.getByText('Target section').closest('[data-slot="bots-section"]') as HTMLElement
    vi.mocked(window.document.elementFromPoint).mockReturnValue(zone)

    fireEvent.pointerDown(row, { button: 0, clientX: 0, clientY: 0, isPrimary: true, pointerId: 1 })
    fireEvent.pointerMove(row, { clientX: 20, clientY: 20, pointerId: 1 })
    expect($draggingBot.get()).toBe('group:Room')
    fireEvent.pointerMove(row, { clientX: 30, clientY: 30, pointerId: 1 })
    fireEvent.pointerUp(row, { clientX: 30, clientY: 30, pointerId: 1 })
    fireEvent.click(row)

    expect(onDropBot).toHaveBeenCalledWith('group:Room')
    expect(onOpen).not.toHaveBeenCalled()
    expect($draggingBot.get()).toBeNull()
  })
})

describe('pointer drop zone registry', () => {
  it('removes an unmounted zone from hit-testing', () => {
    render(
      <SectionDropZone isSource={false} onDropBot={vi.fn()}>
        <div>Target section</div>
      </SectionDropZone>
    )
    const zone = screen.getByText('Target section').closest('[data-slot="bots-section"]') as HTMLElement
    vi.mocked(window.document.elementFromPoint).mockReturnValue(zone)
    expect(findPointerDropZone(10, 10)).not.toBeNull()

    cleanup()

    expect(findPointerDropZone(10, 10)).toBeNull()
  })
})
