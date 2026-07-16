import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $browserCapture, $browserState, BROWSER_QC_DIMENSIONS, type BrowserQc, type BrowserTab } from './store'

const commands = vi.hoisted(() => ({
  back: vi.fn(),
  capture: vi.fn().mockResolvedValue(null),
  forward: vi.fn(),
  navigate: vi.fn(),
  reload: vi.fn(),
  save: vi.fn().mockResolvedValue({ canceled: false, path: '/tmp/capture.png' })
}))

vi.mock('./persistent', () => ({
  BrowserSlot: () => <div data-testid="browser-slot" />,
  browserBack: commands.back,
  browserForward: commands.forward,
  browserNavigate: commands.navigate,
  browserReload: commands.reload,
  captureBrowserTab: commands.capture,
  saveBrowserCapture: commands.save
}))

import { BrowserPane } from './pane'
import { BrowserQcPane } from './qc-pane'

const emptyQc = (): BrowserQc =>
  Object.fromEntries(
    BROWSER_QC_DIMENSIONS.map(dimension => [dimension, { evidence: '', note: '', status: 'unchecked' }])
  ) as BrowserQc

const tab = (overrides: Partial<BrowserTab> = {}): BrowserTab => ({
  id: 'tab-1',
  pinned: false,
  qc: emptyQc(),
  title: 'Fixture',
  url: 'https://example.test/fixture',
  ...overrides
})

beforeEach(() => {
  commands.back.mockClear()
  commands.capture.mockClear()
  commands.forward.mockClear()
  commands.navigate.mockClear()
  commands.reload.mockClear()
  commands.save.mockReset().mockResolvedValue({ canceled: false, path: '/tmp/capture.png' })
  $browserCapture.set(null)
  $browserState.set({ activeTabId: 'tab-1', qcOpen: false, tabs: [tab()] })
})

afterEach(cleanup)

const showCapture = () => {
  act(() => {
    $browserCapture.set({
      captureId: 'capture-1',
      createdAt: 123,
      dataUrl: 'data:image/png;base64,AAAA',
      height: 720,
      tabId: 'tab-1',
      width: 1280
    })
  })
}

describe('BrowserPane', () => {
  it('dispatches navigation only after an explicit form submit', () => {
    render(<BrowserPane />)

    expect(commands.navigate).not.toHaveBeenCalled()
    const input = screen.getByRole('textbox', { name: 'Enter a URL' })
    fireEvent.change(input, { target: { value: 'https://example.test/next' } })
    fireEvent.submit(input.closest('form')!)

    expect(commands.navigate).toHaveBeenCalledWith('tab-1', 'https://example.test/next')
  })

  it('opens QC as a separate pane instead of rendering the inspector inline', () => {
    render(<BrowserPane />)

    expect(screen.queryByRole('group', { name: 'Composition' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'Quality control' }))

    expect($browserState.get().qcOpen).toBe(true)
    expect(screen.queryByRole('group', { name: 'Composition' })).toBeNull()
  })
})

describe('BrowserQcPane', () => {
  it('records manual QC status, note, and evidence per dimension', () => {
    render(<BrowserQcPane />)

    fireEvent.click(screen.getAllByRole('button', { name: 'Fail' })[0])
    fireEvent.change(screen.getAllByRole('textbox', { name: 'Note' })[0], {
      target: { value: 'Subject is off-center' }
    })
    fireEvent.change(screen.getAllByRole('textbox', { name: 'Evidence' })[0], { target: { value: 'fixture capture' } })

    const composition = $browserState.get().tabs[0].qc.composition
    expect(composition).toEqual({ status: 'fail', note: 'Subject is off-center', evidence: 'fixture capture' })
  })
  it('preserves spaces in controlled QC fields while typing', () => {
    render(<BrowserQcPane />)

    const [note] = screen.getAllByRole('textbox', { name: 'Note' })
    const [evidence] = screen.getAllByRole('textbox', { name: 'Evidence' })

    fireEvent.change(note, { target: { value: 'Subject' } })
    fireEvent.change(note, { target: { value: 'Subject ' } })
    expect((note as HTMLInputElement).value).toBe('Subject ')
    expect($browserState.get().tabs[0].qc.composition.note).toBe('Subject ')

    fireEvent.change(note, { target: { value: ' Subject centered ' } })
    fireEvent.change(evidence, { target: { value: 'Fixture' } })
    fireEvent.change(evidence, { target: { value: 'Fixture ' } })
    expect((evidence as HTMLInputElement).value).toBe('Fixture ')
    expect($browserState.get().tabs[0].qc.composition.evidence).toBe('Fixture ')

    expect((note as HTMLInputElement).value).toBe(' Subject centered ')
    expect($browserState.get().tabs[0].qc.composition.note).toBe(' Subject centered ')
  })
})

describe('BrowserPane capture', () => {
  it('does not capture or open a surface from render/background state', () => {
    render(<BrowserPane />)

    expect(commands.capture).not.toHaveBeenCalled()
    fireEvent.click(screen.getByRole('button', { name: 'Capture page' }))
    expect(commands.capture).toHaveBeenCalledWith('tab-1')
  })

  it('keeps external open and capture unavailable for an empty tab', () => {
    $browserState.set({ activeTabId: 'tab-1', qcOpen: false, tabs: [tab({ title: '', url: '' })] })
    render(<BrowserPane />)

    expect((screen.getByRole('button', { name: 'Open externally' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Capture page' }) as HTMLButtonElement).disabled).toBe(true)
    expect(screen.queryByText('Enter a URL to start browsing')).not.toBeNull()
  })

  it('renders a transient capture preview and saves by capture token', async () => {
    render(<BrowserPane />)
    showCapture()

    fireEvent.click(await screen.findByRole('button', { name: 'Save capture' }))
    expect(commands.save).toHaveBeenCalledWith('capture-1')
    await waitFor(() => expect($browserCapture.get()).toBeNull())
  })

  it('retains a capture preview when saving is canceled or fails', async () => {
    render(<BrowserPane />)
    showCapture()
    commands.save.mockResolvedValueOnce({ canceled: true })

    fireEvent.click(await screen.findByRole('button', { name: 'Save capture' }))
    await waitFor(() => expect(commands.save).toHaveBeenCalledWith('capture-1'))
    expect($browserCapture.get()?.captureId).toBe('capture-1')

    commands.save.mockRejectedValueOnce(new Error('disk unavailable'))
    fireEvent.click(screen.getByRole('button', { name: 'Save capture' }))
    await waitFor(() => expect(commands.save).toHaveBeenCalledTimes(2))
    expect($browserCapture.get()?.captureId).toBe('capture-1')
  })
})
