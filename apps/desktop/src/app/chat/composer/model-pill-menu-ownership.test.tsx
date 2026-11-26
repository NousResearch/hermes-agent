import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'

import { DropdownMenuItem, DropdownMenuSearch } from '@/components/ui/dropdown-menu'
import { useShortViewport } from '@/lib/short-viewport'
import { $touchPointer } from '@/lib/touch-interaction'

import { requestModelMenuToggle } from './focus'
import { $composerModelMenuOpen, ModelPill, useComposerModelMenuOpen } from './model-pill'
import { type ComposerScope, ComposerScopeProvider, MAIN_COMPOSER_SCOPE, useComposerScope } from './scope'
import { shortHeightCollapse } from './short-height'
import type { ChatBarState } from './types'

let short = false
const listeners = new Set<() => void>()

const MAIN: ComposerScope = { ...MAIN_COMPOSER_SCOPE, target: 'main' }
const TILE: ComposerScope = { ...MAIN_COMPOSER_SCOPE, target: 'tile:neighbour' }

const MODEL: ChatBarState['model'] = {
  canSwitch: true,
  model: 'gpt-6.1-sol-900k',
  modelMenuContent: (
    <>
      <DropdownMenuSearch placeholder="Find a model" />
      <DropdownMenuItem onSelect={event => event.preventDefault()}>Model settings</DropdownMenuItem>
    </>
  ),
  provider: 'openai-codex'
}

/** One composer's control row at fine pointer, collapsed exactly as the ChatBar collapses it. */
function ShortHeightComposer() {
  const scope = useComposerScope()
  const minimal = shortHeightCollapse(useShortViewport(), false, useComposerModelMenuOpen(scope.target))

  return minimal ? null : <ModelPill compact disabled={false} model={MODEL} />
}

function setShort(next: boolean) {
  short = next
  act(() => listeners.forEach(listener => listener()))
}

afterEach(() => {
  cleanup()
  $touchPointer.set(false)
  $composerModelMenuOpen.set(new Set())
  listeners.clear()
  short = false
  vi.unstubAllGlobals()
})

function stubLayoutHeight() {
  vi.stubGlobal('matchMedia', (query: string) => ({
    get matches() {
      return query === '(max-height: 520px)' && short
    },
    media: query,
    addEventListener: (_type: string, listener: () => void) => {
      if (query === '(max-height: 520px)') {
        listeners.add(listener)
      }
    },
    removeEventListener: (_type: string, listener: () => void) => listeners.delete(listener)
  }))
}

it('keeps a menu opened by the hotkey mounted when the viewport turns short', async () => {
  stubLayoutHeight()
  const surface = document.createElement('div')

  surface.dataset.composerTarget = 'main'
  document.body.append(surface)

  try {
    render(
      <ComposerScopeProvider value={MAIN}>
        <ShortHeightComposer />
      </ComposerScopeProvider>
    )
    const pill = screen.getByRole('button', { name: /^Model ·/ })

    expect($composerModelMenuOpen.get().has('main')).toBe(false)
    await act(async () => {
      requestModelMenuToggle()
      await new Promise(resolve => setTimeout(resolve, 0))
    })
    expect(await screen.findByText('Model settings')).toBeTruthy()
    // The hotkey claims the menu for its composer, so the collapse rule below can see it at all.
    expect($composerModelMenuOpen.get().has('main')).toBe(true)

    setShort(true)

    expect(pill.isConnected).toBe(true)
    expect(screen.getByText('Model settings')).toBeTruthy()
  } finally {
    surface.remove()
  }
})

it('collapses only the composer whose own menu is closed', async () => {
  stubLayoutHeight()
  render(
    <>
      <ComposerScopeProvider value={MAIN}>
        <ShortHeightComposer />
      </ComposerScopeProvider>
      <ComposerScopeProvider value={TILE}>
        <ShortHeightComposer />
      </ComposerScopeProvider>
    </>
  )

  const [open, neighbour] = screen.getAllByRole('button', { name: /^Model ·/ })
  fireEvent.pointerDown(open, { button: 0, pointerType: 'mouse' })
  expect(await screen.findByText('Model settings')).toBeTruthy()

  setShort(true)

  expect(open.isConnected).toBe(true)
  expect(screen.getByText('Model settings')).toBeTruthy()
  expect(neighbour.isConnected).toBe(false)
})

it('unmounting one pill leaves another composer’s menu open', async () => {
  stubLayoutHeight()

  const { rerender } = render(
    <>
      <ComposerScopeProvider key="main" value={MAIN}>
        <ShortHeightComposer />
      </ComposerScopeProvider>
      <ComposerScopeProvider key="tile" value={TILE}>
        <ShortHeightComposer />
      </ComposerScopeProvider>
    </>
  )

  const [, neighbour] = screen.getAllByRole('button', { name: /^Model ·/ })
  fireEvent.pointerDown(neighbour, { button: 0, pointerType: 'mouse' })
  expect(await screen.findByText('Model settings')).toBeTruthy()

  rerender(
    <ComposerScopeProvider key="tile" value={TILE}>
      <ShortHeightComposer />
    </ComposerScopeProvider>
  )

  expect(screen.getByText('Model settings')).toBeTruthy()
  expect($composerModelMenuOpen.get().has('tile:neighbour')).toBe(true)
})
