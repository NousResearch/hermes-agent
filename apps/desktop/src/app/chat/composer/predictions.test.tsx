import { act, cleanup, fireEvent, render } from '@testing-library/react'
import { useRef } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { $activeGatewayProfile } from '@/store/profile'
import { $activeSessionId, $connection } from '@/store/session'

import { useComposerPredictions } from './hooks/use-composer-predictions'
import { ComposerPredictionPreview } from './prediction-preview'
import { $composerPredictions, clearComposerPredictions, setComposerPredictions } from './predictions'
import { composerPlainText, renderComposerContents } from './rich-editor'
import { ComposerVisibleContext } from './visibility'

// Exercise the production hook and overlay through DOM events, not a copied
// key-handler implementation. The surrounding draft/submit ports stay explicit.
function Harness({
  sessionId = 'session-a',
  target = 'main',
  busy = false,
  inputDisabled = false,
  hasTrigger = false,
  hasAttachments = false,
  editingQueue = false,
  hasText = false,
  onSubmit = vi.fn(),
  onAccept = vi.fn(),
  onFallback = vi.fn()
}: {
  sessionId?: string
  target?: 'main' | 'tile:other'
  busy?: boolean
  inputDisabled?: boolean
  hasTrigger?: boolean
  hasAttachments?: boolean
  editingQueue?: boolean
  hasText?: boolean
  onSubmit?: (text: string) => void
  onAccept?: (text: string) => void
  onFallback?: (key: string) => void
}) {
  const editorRef = useRef<HTMLDivElement>(null)
  const draftRef = useRef('')
  const composingRef = useRef(false)

  const predictions = useComposerPredictions({
    sessionId,
    target,
    busy,
    inputDisabled,
    hasTrigger,
    hasAttachments,
    editingQueue,
    hasText,
    editorRef,
    draftRef,
    composingRef,
    insertText: text => {
      draftRef.current = text
      renderComposerContents(editorRef.current!, text)
      onAccept(text)
    }
  })

  return (
    <form
      onSubmit={event => {
        event.preventDefault()
        onSubmit(draftRef.current)
      }}
    >
      <div
        contentEditable
        data-slot="composer-rich-input"
        onCompositionEnd={() => {
          composingRef.current = false
        }}
        onCompositionStart={() => {
          composingRef.current = true
          predictions.invalidate()
        }}
        onInput={predictions.invalidate}
        onKeyDown={event => {
          if (predictions.handleKeyDown(event)) {
            return
          }

          onFallback(event.key)
        }}
        ref={editorRef}
        role="textbox"
        suppressContentEditableWarning
        tabIndex={0}
      />
      {predictions.preview && <ComposerPredictionPreview prediction={predictions.preview} stacked={false} />}
      <button type="submit">Send</button>
    </form>
  )
}

const publish = (suggestions = ['First full message', 'Second\n完整消息', 'Third'], sessionId = 'session-a') =>
  act(() => setComposerPredictions({ sessionId, suggestions, profile: 'default', connectionId: 'local' }))

beforeEach(() => {
  $activeGatewayProfile.set('default')
  $connection.set({
    baseUrl: 'http://localhost',
    wsUrl: 'ws://localhost',
    token: '',
    logs: [],
    nativeOverlayWidth: 0,
    windowButtonPosition: null,
    isFullscreen: false,
    connectionId: 'local'
  })
  $activeSessionId.set('session-a')
})
afterEach(() => {
  cleanup()
  $activeSessionId.set(null)
})

describe('native next-message predictions', () => {
  it('previews full text outside the draft, cycles and wraps, accepts without submitting', () => {
    const onSubmit = vi.fn()
    const onAccept = vi.fn()
    const view = render(<Harness onAccept={onAccept} onSubmit={onSubmit} />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()
    expect(view.getByText('First full message')).toBeTruthy()
    expect(view.getByText('1/3')).toBeTruthy()
    expect(composerPlainText(editor)).toBe('')
    expect(editor.contains(view.getByText('First full message'))).toBe(false)
    expect(fireEvent.keyDown(editor, { key: 'Tab' })).toBe(false)
    expect(view.getByText('2/3')).toBeTruthy()
    expect(view.container.querySelector('[data-slot="composer-prediction-text"]')?.textContent).toBe('Second\n完整消息')
    fireEvent.keyDown(editor, { key: 'Tab' })
    expect(view.getByText('3/3')).toBeTruthy()
    fireEvent.keyDown(editor, { key: 'Tab' })
    expect(view.getByText('1/3')).toBeTruthy()
    fireEvent.keyDown(editor, { key: 'ArrowRight' })
    expect(composerPlainText(editor)).toBe('First full message')
    expect(onAccept).toHaveBeenCalledWith('First full message')
    expect(onSubmit).not.toHaveBeenCalled()
    expect($composerPredictions.get()).toBe(null)
    expect(editor.getAttribute('contenteditable')).toBe('true')
  })

  it('dismisses on Escape and ignores a late result until the next turn', () => {
    const view = render(<Harness />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()
    expect(fireEvent.keyDown(editor, { key: 'Escape' })).toBe(false)
    publish(['Late response'])
    expect($composerPredictions.get()).toBe(null)
    view.rerender(<Harness busy />)
    view.rerender(<Harness />)
    publish(['New turn'])
    expect(view.getByText('New turn')).toBeTruthy()
  })

  it('invalidates synchronously on typing even before the draft flush, including type-then-erase', () => {
    const onAccept = vi.fn()
    const view = render(<Harness onAccept={onAccept} />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()
    act(() => {
      editor.textContent = 'My draft'
      fireEvent.input(editor)
      setComposerPredictions({
        sessionId: 'session-a',
        suggestions: ['Late async result'],
        profile: 'default',
        connectionId: 'local'
      })
      fireEvent.keyDown(editor, { key: 'ArrowRight' })
    })
    expect(composerPlainText(editor)).toBe('My draft')
    expect(onAccept).not.toHaveBeenCalled()
    editor.textContent = ''
    fireEvent.input(editor)
    publish()
    expect($composerPredictions.get()).toBe(null)
  })

  it('checks live DOM at publication and acceptance, not only render-time hasText', () => {
    const onAccept = vi.fn()
    const view = render(<Harness onAccept={onAccept} />)
    const editor = view.getByRole('textbox')
    editor.focus()
    editor.textContent = ' '
    publish()
    expect($composerPredictions.get()).toBe(null)
    editor.textContent = ''
    publish()
    editor.textContent = 'Draft raced ahead of React'
    expect(fireEvent.keyDown(editor, { key: 'ArrowRight' })).toBe(true)
    expect(onAccept).not.toHaveBeenCalled()
    expect(editor.textContent).toBe('Draft raced ahead of React')
  })

  it('invalidates on IME start and never handles composing or legacy 229 keys', () => {
    const view = render(<Harness />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()
    expect(fireEvent.keyDown(editor, { key: 'ArrowRight', isComposing: true })).toBe(true)
    expect(fireEvent.keyDown(editor, { key: 'Tab', keyCode: 229 })).toBe(true)
    fireEvent.compositionStart(editor)
    expect($composerPredictions.get()).toBe(null)
    publish()
    expect(fireEvent.keyDown(editor, { key: 'Tab' })).toBe(true)
    editor.textContent = '你好'
    fireEvent.input(editor)
    fireEvent.compositionEnd(editor)
    expect(editor.textContent).toBe('你好')
    expect($composerPredictions.get()).toBe(null)
  })

  it.each(['shiftKey', 'ctrlKey', 'metaKey', 'altKey'] as const)('preserves %s navigation and shortcuts', modifier => {
    const view = render(<Harness />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()

    for (const key of ['Tab', 'ArrowRight', 'Escape']) {
      expect(fireEvent.keyDown(editor, { key, [modifier]: true })).toBe(true)
    }

    expect(view.getByText('1/3')).toBeTruthy()
    expect(composerPlainText(editor)).toBe('')
  })

  it('does not capture unfocused input, Enter, or other arrow navigation', () => {
    const onFallback = vi.fn()
    const view = render(<Harness onFallback={onFallback} />)
    const editor = view.getByRole('textbox')
    publish()

    for (const key of ['Tab', 'ArrowRight', 'Escape']) {
      expect(fireEvent.keyDown(editor, { key })).toBe(true)
    }

    editor.focus()

    for (const key of ['Enter', 'ArrowUp', 'ArrowDown', 'ArrowLeft']) {
      expect(fireEvent.keyDown(editor, { key })).toBe(true)
      expect(onFallback).toHaveBeenCalledWith(key)
    }

    expect(view.getByText('1/3')).toBeTruthy()
  })

  it.each(['busy', 'inputDisabled', 'hasTrigger', 'hasAttachments', 'editingQueue', 'hasText'] as const)(
    'gates publication and key handling while %s',
    flag => {
      const view = render(<Harness />)
      const editor = view.getByRole('textbox')
      editor.focus()
      publish()
      view.rerender(<Harness {...{ [flag]: true }} />)
      expect(view.queryByText('First full message')).toBe(null)
      publish()
      expect($composerPredictions.get()).toBe(null)

      for (const key of ['Tab', 'ArrowRight', 'Escape']) {
        expect(fireEvent.keyDown(editor, { key })).toBe(true)
      }
    }
  )

  it('scopes results and clears immediately on session switch, without reviving old previews', () => {
    const view = render(<Harness />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()
    act(() => {
      $activeSessionId.set('session-b')
    })
    expect($composerPredictions.get()).toBe(null)
    publish(['Late A result'])
    expect($composerPredictions.get()).toBe(null)
    view.rerender(<Harness sessionId="session-b" />)
    publish(['B result'], 'session-b')
    expect(view.getByText('B result')).toBeTruthy()
    expect(view.getByText('B result')).toBeTruthy()
    act(() => {
      $activeSessionId.set('session-a')
    })
    view.rerender(<Harness />)
    expect(view.queryByText('First full message')).toBe(null)
  })

  it('only binds the main composer and drops predictions on unmount', () => {
    const tile = render(<Harness target="tile:other" />)
    publish()
    expect($composerPredictions.get()).toBe(null)
    tile.unmount()
    const main = render(<Harness />)
    publish()
    expect($composerPredictions.get()).not.toBe(null)
    main.unmount()
    publish()
    expect($composerPredictions.get()).toBe(null)
  })

  it('filters empty/duplicate suggestions without truncating long multiline text', () => {
    const view = render(<Harness />)
    const long = '完整建议'.repeat(200) + '\nhttps://example.test/' + 'x'.repeat(400)
    publish(['  ', long, long, 'Other'])
    expect($composerPredictions.get()?.suggestions).toEqual([long, 'Other'])
    const preview = view.container.querySelector('[data-slot="composer-prediction-text"]')!
    expect(preview.textContent).toBe(long)
    expect(preview.className).not.toMatch(/truncate|line-clamp/)
    act(() => clearComposerPredictions($composerPredictions.get()!.handle))
    expect(view.container.querySelector('[data-slot="composer-prediction-text"]')).toBe(null)
  })

  it.each(['profile', 'connection'] as const)('rebinds after %s flips with the same session id', edge => {
    const onAccept = vi.fn()
    const view = render(<Harness onAccept={onAccept} />)
    const editor = view.getByRole('textbox')
    editor.focus()
    publish()
    const oldHandle = $composerPredictions.get()!.handle
    act(() => {
      if (edge === 'profile') {
        $activeGatewayProfile.set('other')
      } else {
        $connection.set({ ...$connection.get()!, connectionId: 'remote' })
      }

      // Assert before React has flushed its new binding.
      expect($composerPredictions.get()).toBeNull()
      expect(
        setComposerPredictions({
          sessionId: 'session-a',
          profile: 'default',
          connectionId: 'local',
          suggestions: ['Stale']
        })
      ).toBeNull()
      fireEvent.keyDown(editor, { key: 'ArrowRight' })
    })
    expect(onAccept).not.toHaveBeenCalled()
    publish(['Still stale'])
    expect($composerPredictions.get()).toBeNull()
    act(() => {
      expect(
        setComposerPredictions({
          sessionId: 'session-a',
          profile: $activeGatewayProfile.get(),
          connectionId: $connection.get()!.connectionId!,
          suggestions: ['New owner']
        })
      ).not.toBeNull()
      expect(clearComposerPredictions(oldHandle)).toBe(false)
    })
    expect(view.getByText('New owner')).toBeTruthy()
    fireEvent.keyDown(editor, { key: 'ArrowRight' })
    expect(onAccept).toHaveBeenCalledWith('New owner')
  })

  it('rebinds after a batched A→B→A transition without reviving the old preview', () => {
    const view = render(<Harness />)
    publish()
    act(() => {
      $activeGatewayProfile.set('other')
      $activeGatewayProfile.set('default')
      expect($composerPredictions.get()).toBeNull()
      expect(
        setComposerPredictions({
          sessionId: 'session-a',
          profile: 'default',
          connectionId: 'local',
          suggestions: ['Too early']
        })
      ).toBeNull()
    })
    publish(['Fresh binding'])
    expect(view.getByText('Fresh binding')).toBeTruthy()
  })

  it('does not let a superseded mounted receiver dismiss or accept a newer preview', () => {
    const oldAccept = vi.fn()
    const old = render(<Harness onAccept={oldAccept} />)
    publish()
    const current = render(<Harness />)
    publish(['Replacement receiver'])
    const oldEditor = old.getAllByRole('textbox')[0]
    oldEditor.focus()
    fireEvent.keyDown(oldEditor, { key: 'ArrowRight' })
    fireEvent.input(oldEditor)
    expect(oldAccept).not.toHaveBeenCalled()
    expect($composerPredictions.get()?.suggestions).toEqual(['Replacement receiver'])
    old.unmount()
    expect(current.getByText('Replacement receiver')).toBeTruthy()
  })

  it('does not let a hidden keep-alive main composer claim or clear the visible preview', () => {
    const main = render(<Harness />)
    publish()

    const hidden = render(
      <ComposerVisibleContext.Provider value={false}>
        <Harness />
      </ComposerVisibleContext.Provider>
    )

    expect(main.getByText('First full message')).toBeTruthy()
    expect(hidden.container.querySelector('[data-slot="composer-prediction-text"]')).toBeNull()
    publish(['Still visible'])
    expect(main.getByText('Still visible')).toBeTruthy()
    hidden.unmount()
    expect(main.getByText('Still visible')).toBeTruthy()
  })
})
