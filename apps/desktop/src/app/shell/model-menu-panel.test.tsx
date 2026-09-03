import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { useState } from 'react'
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest'

import { useModelControls } from '@/app/session/hooks/use-model-controls'
import { DropdownMenu, DropdownMenuContent } from '@/components/ui/dropdown-menu'
import { $collapsedProviders, toggleCollapsedProvider } from '@/store/provider-collapse'
import { $activeSessionId, $currentModel, $currentProvider } from '@/store/session'

import { ModelMenuPanel } from './model-menu-panel'

const notify = vi.fn((..._args: unknown[]) => 'confirm-toast-1')
const notifyError = vi.fn((..._args: unknown[]) => undefined)
const dismissNotification = vi.fn((..._args: unknown[]) => undefined)

vi.mock('@/store/notifications', () => ({
  dismissNotification: (...args: unknown[]) => dismissNotification(...args),
  notify: (...args: unknown[]) => notify(...args),
  notifyError: (...args: unknown[]) => notifyError(...args)
}))

// Radix calls these on open; jsdom doesn't implement them.
beforeAll(() => {
  Element.prototype.scrollIntoView = vi.fn()
  Element.prototype.hasPointerCapture = vi.fn(() => false)
  Element.prototype.releasePointerCapture = vi.fn()
})

const getGlobalModelOptions = vi.fn()

vi.mock('@/hermes', () => ({
  getGlobalModelOptions: (...args: unknown[]) => getGlobalModelOptions(...args),
  setApiRequestProfile: vi.fn()
}))

// MoA presets now arrive as the catalog's virtual `moa` provider row (the same
// payload a remote gateway's model.options returns), not the /api/model/moa
// REST config.
const MOA_PROVIDER = { models: ['default', 'BeastMode'], name: 'Mixture of Agents', slug: 'moa' }

const DEEPSEEK_PROVIDER = {
  models: ['deepseek-v4-pro', 'deepseek-chat', 'deepseek-reasoner'],
  name: 'DeepSeek',
  slug: 'deepseek'
}

const GOOGLE_PROVIDER = {
  models: ['gemini-3.1-pro', 'gemini-2.5-flash', 'gemini-2.5-pro'],
  name: 'Google',
  slug: 'google'
}

const MOCK_PROVIDERS = [DEEPSEEK_PROVIDER, GOOGLE_PROVIDER, MOA_PROVIDER]

beforeEach(() => {
  $activeSessionId.set('runtime-1')
  $currentModel.set('')
  $currentProvider.set('')
  $collapsedProviders.set([])
  getGlobalModelOptions.mockResolvedValue({ providers: MOCK_PROVIDERS })
})

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

function renderPanel(onSelectModel = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })

  const requestGateway = vi.fn(async (method: string) => {
    if (method === 'model.options') {
      return getGlobalModelOptions()
    }

    throw new Error(`unexpected gateway method: ${method}`)
  })

  const content = render(
    <QueryClientProvider client={client}>
      <DropdownMenu open>
        <DropdownMenuContent>
          <ModelMenuPanel onSelectModel={onSelectModel} requestGateway={requestGateway as never} />
        </DropdownMenuContent>
      </DropdownMenu>
    </QueryClientProvider>
  )

  return { onSelectModel, content }
}

describe('ModelMenuPanel MoA presets', () => {
  it('selecting a MoA preset switches PERSISTENTLY via onSelectModel (not the one-shot dispatch)', async () => {
    const { content, onSelectModel } = renderPanel()

    // moaOptions is async (useQuery) — wait for the preset row to mount.
    const row = await content.findByText('MoA: BeastMode')
    fireEvent.click(row)

    // #54670: must route through the persistent model-switch path
    // i.e. onSelectModel with provider 'moa' (which session-scopes live-session
    // switches), NOT a one-shot command.dispatch that reverts after a turn.
    expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa', sessionId: 'runtime-1' })
  })

  it('shows the check on the preset that matches the current moa selection', async () => {
    $currentProvider.set('moa')
    $currentModel.set('BeastMode')
    const { content } = renderPanel()

    const row = await content.findByText('MoA: BeastMode')
    // The check codicon renders as a sibling within the same row item.
    const item = row.closest('[role="menuitem"]') ?? row.parentElement
    expect(item?.querySelector('.codicon-check')).not.toBeNull()
  })

  it('keeps the virtual moa provider out of the main model groups (presets section only)', async () => {
    const { content } = renderPanel()

    await content.findByText('MoA: BeastMode')

    // The provider group header would read "Mixture of Agents"; the presets
    // section header reads "MoA presets". Only the latter should exist.
    // Radix DropdownMenu portals its content to document.body, so assert
    // against the body (not content.container) to see the rendered items.

    // eslint-disable-next-line no-restricted-globals
    expect(document.body.textContent).toContain('MoA presets')
    // eslint-disable-next-line no-restricted-globals
    expect(document.body.textContent).not.toContain('Mixture of Agents')
  })

  it('renders presets from the catalog even before a session exists', async () => {
    $activeSessionId.set('')
    const { onSelectModel, content } = renderPanel()

    const row = await content.findByText('MoA: BeastMode')
    fireEvent.click(row)

    // Pre-session picks are UI state shipped on the next session.create — the
    // row must not be disabled and must still route through onSelectModel.
    expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa', sessionId: null })
  })
})

describe('ModelMenuPanel current selection', () => {
  it('keeps the checkmark on the live SessionView model when a stale options response disagrees', async () => {
    $currentProvider.set('google')
    $currentModel.set('gemini-3.1-pro')
    getGlobalModelOptions.mockResolvedValue({
      model: 'deepseek-chat',
      provider: 'deepseek',
      providers: MOCK_PROVIDERS
    })

    const { content } = renderPanel()

    const currentRow = (await content.findByText(/Gemini 3\.1 Pro/i)).closest('[role="menuitem"]')
    const staleRow = content.getByText('Deepseek Chat').closest('[role="menuitem"]')

    expect(currentRow?.querySelector('.codicon-check')).not.toBeNull()
    expect(staleRow?.querySelector('.codicon-check')).toBeNull()
  })
})

describe('ModelMenuPanel search', () => {
  // The pinned current model must NOT ride along on a query it doesn't match:
  // it reads like the top result, so Enter/click picks the wrong model (the
  // "type grok, get fable" bug). Every surveyed picker (VS Code, Zed, Open
  // WebUI, Cherry Studio) drops the pin while filtering.
  // Highlighted labels are split across <mark> nodes, so single-text-node
  // queries miss them — match on the row span's composed textContent.
  const rowWithText = (content: ReturnType<typeof renderPanel>['content'], pattern: RegExp) =>
    content.queryByText((_, element) => element?.tagName === 'SPAN' && pattern.test(element.textContent ?? ''))

  it('hides the non-matching current model while a query is active', async () => {
    $currentProvider.set('deepseek')
    $currentModel.set('deepseek-v4-pro')
    const { content } = renderPanel()

    await content.findByText(/Deepseek V4 Pro/i)

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'gemini' } })

    await vi.waitFor(() => {
      expect(rowWithText(content, /Gemini 3\.1 Pro/i)).not.toBeNull()
    })
    expect(rowWithText(content, /Deepseek V4 Pro/i)).toBeNull()
  })

  it('Enter in the search field commits the first match', async () => {
    const { content, onSelectModel } = renderPanel()

    await content.findByText('DeepSeek')

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'gemini' } })

    await vi.waitFor(() => {
      expect(rowWithText(content, /Gemini 3\.1 Pro/i)).not.toBeNull()
    })

    fireEvent.keyDown(input, { key: 'Enter' })

    // First matching family of the first (alphabetical) matching provider.
    await vi.waitFor(() => {
      expect(onSelectModel).toHaveBeenCalledWith({
        model: 'gemini-3.1-pro',
        provider: 'google',
        sessionId: 'runtime-1'
      })
    })
  })

  it('Enter with no matches is a no-op (menu stays put, nothing selected)', async () => {
    const { content, onSelectModel } = renderPanel()

    await content.findByText('DeepSeek')

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'zzz-no-such-model' } })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onSelectModel).not.toHaveBeenCalled()
  })

  it('arrows move the selection without leaving the input; Enter commits the stepped row', async () => {
    const { content, onSelectModel } = renderPanel()

    await content.findByText('DeepSeek')

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'gemini' } })

    await vi.waitFor(() => {
      expect(rowWithText(content, /Gemini 3\.1 Pro/i)).not.toBeNull()
    })

    // First match auto-selected; ↓ steps to the second match.
    fireEvent.keyDown(input, { key: 'ArrowDown' })
    fireEvent.keyDown(input, { key: 'Enter' })

    await vi.waitFor(() => {
      expect(onSelectModel).toHaveBeenCalledWith({
        model: 'gemini-2.5-flash',
        provider: 'google',
        sessionId: 'runtime-1'
      })
    })
  })

  it('with no query the selection sits on the current model, so Enter closes without switching', async () => {
    $currentProvider.set('google')
    $currentModel.set('gemini-3.1-pro')
    const { content, onSelectModel } = renderPanel()

    await content.findByText('DeepSeek')

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.keyDown(input, { key: 'Enter' })

    expect(onSelectModel).not.toHaveBeenCalled()
  })

  it('filters MoA presets by the query instead of leaving them as phantom first matches', async () => {
    const { content, onSelectModel } = renderPanel()

    await content.findByText('MoA: BeastMode')

    const input = screen.getByRole('textbox', { name: 'Search models' })
    fireEvent.change(input, { target: { value: 'beast' } })

    await vi.waitFor(() => {
      expect(rowWithText(content, /MoA: BeastMode/)).not.toBeNull()
    })
    expect(rowWithText(content, /MoA: default/)).toBeNull()

    // The surviving preset IS the first row, so Enter commits it.
    fireEvent.keyDown(input, { key: 'Enter' })

    await vi.waitFor(() => {
      expect(onSelectModel).toHaveBeenCalledWith({ model: 'BeastMode', provider: 'moa', sessionId: 'runtime-1' })
    })
  })
})

describe('ModelMenuPanel provider collapse', () => {
  it('shows all provider models by default (none collapsed)', async () => {
    const { content } = renderPanel()

    await content.findByText('DeepSeek')
    expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
    expect(content.queryByText('Deepseek Chat')).not.toBeNull()
  })

  it('collapses provider models when header is clicked', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    fireEvent.click(header)

    // Models should disappear but header stays
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
    expect(content.queryByText('DeepSeek')).not.toBeNull()
  })

  it('expands provider models when header is clicked again', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    // Collapse
    fireEvent.click(header)
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
    // Expand
    fireEvent.click(header)
    await vi.waitFor(() => {
      expect(content.queryByText('Deepseek V4 Pro')).not.toBeNull()
    })
  })

  it('collapses the active provider too (no forced auto-expand)', async () => {
    $currentProvider.set('deepseek')
    $currentModel.set('deepseek-v4-pro')
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    fireEvent.click(header)

    // The current provider is collapsible like any other — clicking its header
    // hides its models rather than forcing them to stay open.
    await vi.waitFor(() => {
      expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
    })
  })

  it('bypasses collapse when search is active', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    fireEvent.click(header)
    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()

    // Type in the search bar (auto-focused by DropdownMenuSearch)
    const input = screen.getByRole('textbox', { name: 'Search models' })
    expect(input).not.toBeNull()
    fireEvent.change(input, { target: { value: 'deepseek' } })

    // Should show models — search bypasses collapse. The matched letters render
    // inside a <mark>, splitting the label across nodes, so match on the row
    // span's composed textContent instead of a single text node.
    await vi.waitFor(() => {
      expect(
        content.queryByText(
          (_, element) => element?.tagName === 'SPAN' && (element.textContent ?? '').startsWith('Deepseek V4 Pro')
        )
      ).not.toBeNull()
    })
  })

  it('toggles collapse via keyboard Enter on header', async () => {
    const { content } = renderPanel()

    const header = await content.findByText('DeepSeek')
    // Radix DropdownMenuItem fires onSelect on Enter from the onKeyDown handler
    fireEvent.keyDown(header.closest('[role="menuitem"]') ?? header, { key: 'Enter' })

    expect(content.queryByText('Deepseek V4 Pro')).toBeNull()
  })

  // The collapsed-providers set is a global presentation preference
  // (`hermes.desktop.collapsed-providers`), but the catalog the picker renders
  // is profile-scoped (`getGlobalModelOptions` routes through
  // `profileScoped()`). Pruning the global set against only the active catalog
  // would silently delete a user's collapse preference on every profile switch
  // whose configured providers don't include the slug — the bug the maintainer
  // flagged. The set must survive catalog changes; if the same provider shows
  // up again later, the previous collapse is preserved.
  it('preserves the collapsed set across a profile switch whose catalog lacks the slug', async () => {
    toggleCollapsedProvider('deepseek')
    toggleCollapsedProvider('google')
    expect($collapsedProviders.get()).toEqual(['deepseek', 'google'])

    // Profile A: both providers present, render + unmount.
    getGlobalModelOptions.mockResolvedValueOnce({ providers: MOCK_PROVIDERS })
    const a = renderPanel()
    await a.content.findByText('DeepSeek')
    a.content.unmount()

    // Profile B: google is not in the catalog (simulates a profile whose
    // configured providers differ). The previously-collapsed 'google' slug
    // must survive — pruning it would lose state across a profile switch.
    getGlobalModelOptions.mockResolvedValueOnce({ providers: [DEEPSEEK_PROVIDER, MOA_PROVIDER] })
    const b = renderPanel()
    await b.content.findByText('DeepSeek')

    expect($collapsedProviders.get()).toEqual(['deepseek', 'google'])
  })

  it('preserves the collapsed set when Refresh Models drops a provider', async () => {
    toggleCollapsedProvider('deepseek')
    toggleCollapsedProvider('google')

    // First load: both providers present.
    getGlobalModelOptions.mockResolvedValueOnce({ providers: MOCK_PROVIDERS })
    const a = renderPanel()
    await a.content.findByText('DeepSeek')
    a.content.unmount()

    // Refresh Models returns a catalog that drops google (revoked key,
    // plugin disabled, backend policy change). 'google' must survive — the
    // user explicitly collapsed it, and the global set is not tied to any
    // single refresh.
    getGlobalModelOptions.mockResolvedValueOnce({ providers: [DEEPSEEK_PROVIDER, MOA_PROVIDER] })
    const b = renderPanel()
    await b.content.findByText('DeepSeek')

    expect($collapsedProviders.get()).toContain('google')
    expect($collapsedProviders.get()).toContain('deepseek')
  })

  it('switches the session model when Refresh Models drops the current pick', async () => {
    $currentProvider.set('zhipu')
    $currentModel.set('glm-4.5-air')
    getGlobalModelOptions
      .mockResolvedValueOnce({
        model: 'glm-4.5-air',
        provider: 'zhipu',
        providers: [{ models: ['glm-4.5-air', 'glm-5-turbo'], name: '智谱2', slug: 'zhipu' }, MOA_PROVIDER]
      })
      .mockResolvedValueOnce({
        model: 'glm-4.5-air',
        provider: 'zhipu',
        providers: [DEEPSEEK_PROVIDER, MOA_PROVIDER]
      })

    const { content, onSelectModel } = renderPanel()

    await content.findByText(/Glm 4\.5 Air/i)

    fireEvent.click(await content.findByText('Refresh models'))

    await vi.waitFor(() => {
      expect(onSelectModel).toHaveBeenCalledWith({
        model: 'deepseek-v4-pro',
        provider: 'deepseek',
        sessionId: 'runtime-1'
      })
    })
  })

  it('does not switch when Refresh Models still lists the current pick', async () => {
    $currentProvider.set('deepseek')
    $currentModel.set('deepseek-v4-pro')
    getGlobalModelOptions.mockResolvedValue({ providers: MOCK_PROVIDERS })

    const { content, onSelectModel } = renderPanel()

    await content.findByText(/Deepseek V4 Pro/i)
    fireEvent.click(await content.findByText('Refresh models'))

    await vi.waitFor(() => {
      expect(getGlobalModelOptions).toHaveBeenCalledTimes(2)
    })
    expect(onSelectModel).not.toHaveBeenCalled()
  })
})

describe('ModelMenuPanel refresh reconcile × guarded-switch confirm handshake', () => {
  // #95446 fix (reconcile after Refresh Models) composes with the
  // confirm-handshake guard: when the reconcile target is itself a GUARDED
  // model (contributor tier / expensive), the switch must surface the confirm
  // flow — one config.set, a warning with a Confirm action, rollback until
  // confirmed — never a silent retry loop and never a silently-painted pick.
  function ConfirmHarness({
    requestGateway
  }: {
    requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
  }) {
    const [client] = useState(() => new QueryClient({ defaultOptions: { queries: { retry: false } } }))
    const controls = useModelControls({ queryClient: client, requestGateway })

    return (
      <QueryClientProvider client={client}>
        <DropdownMenu open>
          <DropdownMenuContent>
            <ModelMenuPanel onSelectModel={controls.selectModel} requestGateway={requestGateway as never} />
          </DropdownMenuContent>
        </DropdownMenu>
      </QueryClientProvider>
    )
  }

  it('reconcile-triggered switch to a guarded model surfaces confirm, not a silent retry', async () => {
    $activeSessionId.set('runtime-1')
    $currentProvider.set('zhipu')
    $currentModel.set('glm-4.5-air')
    getGlobalModelOptions
      .mockResolvedValueOnce({
        providers: [{ models: ['glm-4.5-air'], name: 'Zhipu', slug: 'zhipu' }, MOA_PROVIDER]
      })
      // Refresh drops the current pick; the only remaining model is guarded.
      .mockResolvedValueOnce({
        providers: [{ models: ['muse-spark-1.2-contributor'], name: 'OpenCode', slug: 'opencode-go' }, MOA_PROVIDER]
      })

    // Method-aware gateway: the panel's catalog reads (`model.options`) use
    // the routed catalog mock; `config.set` runs the guarded handshake —
    // confirm_required first, success on the confirmed resend.
    let configSets = 0

    const requestGateway = vi.fn(async (method: string, _params?: Record<string, unknown>) => {
      if (method === 'model.options') {
        return getGlobalModelOptions()
      }

      if (method !== 'config.set') {
        throw new Error(`unexpected gateway method: ${method}`)
      }

      configSets += 1

      if (configSets === 1) {
        return {
          confirm_message: 'CONTRIBUTOR TIER: this model may train on your data.',
          confirm_required: true,
          key: 'model',
          value: 'muse-spark-1.2-contributor'
        }
      }

      return { key: 'model', scope: 'global', value: 'muse-spark-1.2-contributor' }
    })

    const content = render(<ConfirmHarness requestGateway={requestGateway as never} />)

    await content.findByText(/Glm 4\.5 Air/i)
    fireEvent.click(await content.findByText('Refresh models'))

    // The reconcile fired exactly ONE switch attempt and it came back
    // confirm_required → the confirm toast is up, nothing retried silently.
    await vi.waitFor(() => {
      expect(notify).toHaveBeenCalledWith(
        expect.objectContaining({
          action: expect.objectContaining({ label: expect.any(String) }),
          kind: 'warning',
          message: 'CONTRIBUTOR TIER: this model may train on your data.'
        })
      )
    })

    const configSetCalls = requestGateway.mock.calls.filter(([method]) => method === 'config.set')
    expect(configSetCalls).toHaveLength(1)
    expect(configSetCalls[0][1]).not.toHaveProperty('confirm_expensive_model')

    // Pending confirmation = rolled back, not silently painted.
    expect($currentModel.get()).toBe('glm-4.5-air')
    expect($currentProvider.get()).toBe('zhipu')

    // User confirms → ONE resend carrying confirm_expensive_model: true.
    const lastNotify = notify.mock.calls.at(-1)?.[0] as { action: { onClick: () => Promise<void> } }

    await act(async () => {
      await lastNotify.action.onClick()
    })

    await vi.waitFor(() => {
      const resend = requestGateway.mock.calls.filter(([method]) => method === 'config.set')
      expect(resend).toHaveLength(2)
      expect(resend[1][1]).toMatchObject({ confirm_expensive_model: true, session_id: 'runtime-1' })
    })
    expect($currentModel.get()).toBe('muse-spark-1.2-contributor')
    expect($currentProvider.get()).toBe('opencode-go')
    expect(notifyError).not.toHaveBeenCalled()
  })
})
