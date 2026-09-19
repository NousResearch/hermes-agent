import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import { atom } from 'nanostores'
import { useContext } from 'react'
import { afterEach, describe, expect, it } from 'vitest'

import type { ChatBarState } from '@/app/chat/composer/types'
import { type SessionView, SessionViewProvider } from '@/app/chat/session-view'
import { ModelMenuCloseContext } from '@/app/shell/model-menu-panel'
import { registry } from '@/contrib/registry'
import { $activeSessionId, $currentModel, setCurrentModel, setCurrentModelSource } from '@/store/session'

import { COMPOSER_AREAS, type ComposerModelPillProvider } from './contrib'
import { requestModelMenuToggle } from './focus'
import { ModelPill } from './model-pill'
import { RICH_INPUT_SLOT } from './rich-editor'

const modelState = (over: Partial<ChatBarState['model']> = {}): ChatBarState['model'] => ({
  canSwitch: true,
  model: 'gpt-6',
  provider: 'openai',
  ...over
})

afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
  setCurrentModel('')
  setCurrentModelSource('')
})

// #62055: a manual composer pick is sticky and silently overrides the
// Settings → Model default for every NEW chat. The pill must say so.
describe('ModelPill pinned-override badge', () => {
  it('shows the pin dot on a draft running a manual pick', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentModelSource('manual')
    $activeSessionId.set(null)

    render(<ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />)

    expect(screen.getByTestId('model-pinned-dot')).toBeTruthy()
  })

  it('stays quiet when the composer reflects the profile default', () => {
    setCurrentModel('google/gemma-4-26b-a4b-it:free')
    setCurrentModelSource('default')
    $activeSessionId.set(null)

    render(<ModelPill disabled={false} model={modelState()} />)

    expect(screen.queryByTestId('model-pinned-dot')).toBeNull()
  })

  it('stays quiet on a live session (footer shows that session, not the pin)', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentModelSource('manual')
    $activeSessionId.set('live-1')

    render(<ModelPill disabled={false} model={modelState()} />)

    expect(screen.queryByTestId('model-pinned-dot')).toBeNull()
  })

  it('is exercised in both render paths', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')
    setCurrentModelSource('manual')
    $activeSessionId.set(null)

    // Fallback (no live menu) path.
    const { unmount } = render(
      <ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />
    )

    expect(screen.getByTestId('model-pinned-dot')).toBeTruthy()
    unmount()

    // Live-menu (dropdown) path.
    render(
      <ModelPill
        disabled={false}
        model={modelState({ model: 'deepseek/deepseek-v4-flash', modelMenuContent: <div /> })}
      />
    )
    expect(screen.getByTestId('model-pinned-dot')).toBeTruthy()
    expect($currentModel.get()).toBe('deepseek/deepseek-v4-flash')
  })
})

function MenuChoice() {
  const close = useContext(ModelMenuCloseContext)

  return <button onClick={() => close?.()}>Choose model</button>
}

it('returns to the exact caret or backward selection after the model menu closes', async () => {
  const surface = document.createElement('div')
  surface.dataset.composerTarget = 'main'
  const editor = document.createElement('div')
  editor.dataset.slot = RICH_INPUT_SLOT
  editor.contentEditable = 'true'
  editor.tabIndex = 0
  editor.textContent = 'before and after'
  surface.append(editor)
  document.body.append(surface)

  try {
    render(<ModelPill disabled={false} model={modelState({ modelMenuContent: <MenuChoice /> })} />)

    for (const [anchor, focus] of [
      [3, 3],
      [10, 4]
    ]) {
      editor.focus()
      window.getSelection()!.setBaseAndExtent(editor.firstChild!, anchor, editor.firstChild!, focus)
      await act(async () => {
        requestModelMenuToggle()
        await new Promise(resolve => setTimeout(resolve, 0))
      })
      const choice = await screen.findByText('Choose model')
      choice.focus()
      window.getSelection()!.removeAllRanges()
      fireEvent.click(choice)
      await waitFor(() => expect(document.activeElement).toBe(editor))
      expect(window.getSelection()!.anchorOffset).toBe(anchor)
      expect(window.getSelection()!.focusOffset).toBe(focus)
    }
  } finally {
    surface.remove()
  }
})

describe('ModelPill per-surface model label', () => {
  it('shows the chat-bar model even when the primary global differs', () => {
    setCurrentModel('primary/model')
    $activeSessionId.set('primary-runtime')

    const tileView: SessionView = {
      kind: 'tile',
      $awaitingResponse: atom(false),
      $busy: atom(false),
      $cwd: atom(''),
      $fast: atom(false),
      $lastVisibleIsUser: atom(false),
      $messages: atom([]),
      $messagesEmpty: atom(true),
      $model: atom('tile/claude-sonnet'),
      $provider: atom('anthropic'),
      $reasoningEffort: atom('high'),
      $reasoningEffortPending: atom(false),
      $reasoningEffortWire: atom(''),
      $runtimeId: atom('tile-runtime'),
      $storedId: atom('stored-tile'),
      $turnStartedAt: atom<number | null>(null)
    }

    render(
      <SessionViewProvider value={tileView}>
        <ModelPill
          disabled={false}
          model={modelState({ model: 'tile/claude-sonnet', provider: 'anthropic', modelMenuContent: <div /> })}
        />
      </SessionViewProvider>
    )

    expect(screen.getByText('Sonnet')).toBeTruthy()
    expect(screen.queryByText(/primary/i)).toBeNull()
  })
})

// The `composer.modelPill` slot: a provider may override the pill's LABEL
// (compact reasoning label, custom naming) while the pill keeps its chrome,
// pin dot, and menu. A declining provider leaves the core label untouched.
describe('ModelPill label providers', () => {
  const disposers: Array<() => void> = []

  afterEach(() => {
    disposers.splice(0).forEach(dispose => dispose())
  })

  const register = (label: ComposerModelPillProvider['label'], id = 'pill-label') =>
    disposers.push(
      registry.register({
        area: COMPOSER_AREAS.modelPill,
        data: { label } satisfies ComposerModelPillProvider,
        id,
        source: 'disk'
      })
    )

  it('renders a provider-supplied label, and the core label once the provider declines', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')

    register(({ model, reasoningEffort }) => `${model} · ${reasoningEffort || 'none'}`)

    const { unmount } = render(
      <ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />
    )

    expect(screen.getByText('deepseek/deepseek-v4-flash · none')).toBeTruthy()
    unmount()

    // Declining provider: the override text is gone, the core label is back.
    disposers.splice(0).forEach(dispose => dispose())
    register(() => null)

    render(<ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />)

    expect(screen.queryByText(/· none/)).toBeNull()
  })

  it('treats a throwing provider as declining and lets the next provider win', () => {
    setCurrentModel('deepseek/deepseek-v4-flash')

    register(() => {
      throw new Error('broken provider')
    }, 'broken')
    register(() => 'next wins', 'working')

    render(<ModelPill disabled={false} model={modelState({ model: 'deepseek/deepseek-v4-flash' })} />)

    // The broken provider declined; the next one's label renders.
    expect(screen.getByText('next wins')).toBeTruthy()
  })
})
