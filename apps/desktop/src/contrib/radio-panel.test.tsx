import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react'
import { atom } from 'nanostores'
import type { ReactElement } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

interface RadioModule {
  Stations: (props: { player: unknown; ctx: unknown }) => ReactElement
  PRESETS: Array<{ id: string; name: string; url: string }>
}

const modules = import.meta.glob<RadioModule>('../plugins/radio/plugin.js', { eager: true })
const { Stations, PRESETS } = Object.values(modules)[0]

const stored = [...PRESETS]
const value = <T,>(initial: T) => atom(initial)
const player = {
  station: value(PRESETS[0]),
  stations: value(stored),
  favorites: value([] as typeof stored),
  status: value('paused'),
  volume: value(25),
  meterMode: value('levels'),
  play: vi.fn(),
  toggle: vi.fn(),
  stop: vi.fn(),
  favorite: vi.fn(),
  updateStation: vi.fn(),
  removeStation: vi.fn(),
  addStation: vi.fn(({ name, url }: { name: string; url: string }) => {
    const item = { id: `custom-${name}`, name, url }
    stored.push(item)
    player.stations.set([...stored])
    return { station: item, added: true }
  })
}

const ctx = {
  rest: vi.fn(async (path: string) => {
    if (path.includes('/shoutcast/')) return { stations: [{ id: 'sc-1', name: 'Smooth Jazz UK', url: 'http://example.test/sc1' }], skipped: 2, count: 40, next: 30 }
    return { stations: [{ id: 'ti-1', name: 'BNR TuneIn', url: 'https://example.test/bnr' }], skipped: 1, count: 40, next: 30 }
  })
}

const json = (data: unknown) => ({ ok: true, json: async () => data })

function renderPanel() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } })
  return render(
    <QueryClientProvider client={client}>
      <Stations player={player} ctx={ctx} />
    </QueryClientProvider>
  )
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn(async (url: string) => {
    const request = new URL(url)
    if (request.pathname.endsWith('/json/servers')) return json([{ name: 'de1.api.radio-browser.info' }])
    return json([
      { stationuuid: 'pop-1', name: 'Popular Station', url_resolved: 'https://example.test/popular', hls: false },
      { stationuuid: 'pop-2', name: 'Christmas FM', url_resolved: 'https://example.test/xmas', hls: false }
    ])
  }))
})

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
  vi.unstubAllGlobals()
})

describe('Radio picker panel', () => {
  it('exposes four keyboard-accessible source chips with one selected', () => {
    renderPanel()
    const chips = [...document.querySelectorAll<HTMLButtonElement>('.hermes-radio-chip')]
    expect(chips.map(chip => chip.textContent)).toEqual(['sourceAll', 'sourcePopular', 'sourceShoutcast', 'sourceTunein'])
    expect(chips.map(chip => chip.getAttribute('aria-pressed'))).toEqual(['true', 'false', 'false', 'false'])
    expect(chips[1].tagName).toBe('BUTTON')
  })

  it('swaps the list when a directory chip is selected and shows skipped rows', async () => {
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'sourceShoutcast' }))
    expect(await screen.findByText('Smooth Jazz UK')).toBeTruthy()
    expect(screen.getByText(/2 skippedNote/)).toBeTruthy()
    expect(screen.queryByText(PRESETS[0].name)).toBeNull()

    fireEvent.click(screen.getByRole('button', { name: 'sourceTunein' }))
    // TuneIn has no top list, so the row appears only once a name is typed.
    fireEvent.change(screen.getByPlaceholderText('search'), { target: { value: 'jazz' } })
    expect(await screen.findByText('BNR TuneIn')).toBeTruthy()
    expect(screen.getByText(/1 skippedNote/)).toBeTruthy()
  })

  it('opens the add form and the stored-row editor with name and address inputs', () => {
    renderPanel()
    fireEvent.click(screen.getByRole('button', { name: 'add' }))
    expect(screen.getByLabelText('addName')).toBeTruthy()
    expect(screen.getByLabelText('addUrl')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'add' }))
    expect(screen.queryByLabelText('addName')).toBeNull()

    const row = [...document.querySelectorAll('.hermes-radio-row')].find(node => node.textContent?.includes(PRESETS[0].name))
    expect(row).toBeTruthy()
    fireEvent.click(within(row as HTMLElement).getByRole('button', { name: `edit: ${PRESETS[0].name}` }))
    expect(screen.getByLabelText('editName')).toBeTruthy()
    expect(screen.getByLabelText('editUrl')).toBeTruthy()
  })
})
