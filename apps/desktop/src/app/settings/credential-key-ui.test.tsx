import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { stubResizeObserver } from '@/test/jsdom'

import { CredentialKeyCard, type KeyRowProps } from './credential-key-ui'
import { envVar } from './test-utils'

stubResizeObserver()

const desktopWindow = window as unknown as { hermesDesktop?: Window['hermesDesktop'] }
const initialHermesDesktop = desktopWindow.hermesDesktop

function installDesktopBridge(partial: Partial<Window['hermesDesktop']> = {}) {
  desktopWindow.hermesDesktop = {
    fetchLinkTitle: vi.fn().mockResolvedValue(''),
    openExternal: vi.fn().mockResolvedValue(undefined),
    ...partial
  } as unknown as Window['hermesDesktop']
}

function rowProps(patch: Partial<KeyRowProps> = {}): KeyRowProps {
  return {
    edits: {},
    onClear: vi.fn(),
    onReveal: vi.fn(),
    onSave: vi.fn(),
    revealed: {},
    saving: null,
    setEdits: vi.fn(),
    ...patch
  }
}

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()

  if (initialHermesDesktop) {
    desktopWindow.hermesDesktop = initialHermesDesktop
  } else {
    delete desktopWindow.hermesDesktop
  }
})

describe('CredentialKeyCard Get-key link', () => {
  it('opens the docs URL through hermesDesktop.openExternal instead of target=_blank', () => {
    const openExternal = vi.fn().mockResolvedValue(undefined)
    const docsUrl = 'https://platform.deepseek.com/api_keys'

    installDesktopBridge({ openExternal: openExternal as unknown as Window['hermesDesktop']['openExternal'] })

    render(
      <CredentialKeyCard
        expanded
        info={envVar('provider', { url: docsUrl })}
        label="DeepSeek"
        onExpand={vi.fn()}
        onToggle={vi.fn()}
        placeholder="Paste key"
        rowProps={rowProps()}
        varKey="DEEPSEEK_API_KEY"
      />
    )

    fireEvent.click(screen.getByRole('link', { name: 'Get a key' }))

    expect(openExternal).toHaveBeenCalledWith(docsUrl)
  })

  it('does not render a Get-key link when the catalog URL is empty', () => {
    render(
      <CredentialKeyCard
        expanded
        info={envVar('provider', { description: 'Provider API key.', url: '' })}
        label="OpenAI"
        onExpand={vi.fn()}
        onToggle={vi.fn()}
        placeholder="Paste key"
        rowProps={rowProps()}
        varKey="OPENAI_API_KEY"
      />
    )

    expect(screen.queryByRole('link', { name: 'Get a key' })).toBeNull()
  })
})
