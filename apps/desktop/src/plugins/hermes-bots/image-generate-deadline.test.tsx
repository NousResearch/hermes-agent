import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { translateBots } from './i18n-test-helper'

// #86161 follow-up: generateAvatarImage got IMAGE_GENERATE_TIMEOUT_MS, but the
// other two image.generate call sites (custom-prompt avatar, group picture)
// kept the socket's generic 30 s deadline and dropped slow renders.

const { host } = vi.hoisted(() => ({
  host: { notifyError: vi.fn(), request: vi.fn() } as Record<string, ReturnType<typeof vi.fn>>
}))

vi.mock('@hermes/plugin-sdk', async () => {
  const { pluginSdkMock } = await import('./group-test-utils')
  const base = await pluginSdkMock(host)
  const Plain = ({ children }: { children?: ReactNode }) => <>{children}</>

  return {
    ...base,
    Button: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    cn: (...values: unknown[]) => values.filter(Boolean).join(' '),
    Codicon: () => null,
    ColorSwatches: () => null,
    GlyphSpinner: () => null,
    PROFILE_SWATCHES: [],
    profileColor: () => undefined,
    RowButton: (props: React.ComponentProps<'button'>) => <button type="button" {...props} />,
    SegmentedControl: ({
      options,
      onChange
    }: {
      onChange: (id: string) => void
      options: { id: string; label: string }[]
    }) => (
      <div>
        {options.map(o => (
          <button key={o.id} onClick={() => onChange(o.id)} type="button">
            {`tab:${o.id}`}
          </button>
        ))}
      </div>
    ),
    Textarea: (props: React.ComponentProps<'textarea'>) => <textarea {...props} />,
    Tip: Plain,
    useI18n: () => ({ t: { common: { remove: 'Remove' } } }),
    usePluginI18n: () => translateBots
  }
})

vi.mock('./pet', () => ({ PetTab: () => null }))

beforeEach(async () => {
  vi.resetModules()
  host.request.mockReset().mockResolvedValue({ success: false, error: 'stop here' })
  host.notifyError.mockReset()
  const { $imagenAvailable } = await import('./avatar-image')
  $imagenAvailable.set(true)
})

afterEach(cleanup)

async function expectDeadline() {
  const { IMAGE_GENERATE_TIMEOUT_MS } = await import('./avatar-image')
  await waitFor(() => expect(host.request).toHaveBeenCalled())
  const [method, , timeoutMs] = host.request.mock.calls[0] as [string, unknown, number]
  expect(method).toBe('image.generate')
  expect(timeoutMs).toBe(IMAGE_GENERATE_TIMEOUT_MS)
  expect(timeoutMs).toBeGreaterThan(30_000)
}

describe('image.generate deadline at the remaining call sites (#86161)', () => {
  it('custom-prompt avatar generation passes IMAGE_GENERATE_TIMEOUT_MS', async () => {
    const { AvatarPicker } = await import('./avatar-picker')
    render(
      <AvatarPicker color={null} image={null} onColor={vi.fn()} onImage={vi.fn()} onShape={vi.fn()} shape="blobatar" />
    )

    fireEvent.click(screen.getByText('tab:generate'))
    fireEvent.change(screen.getByPlaceholderText(translateBots('avatar.describePlaceholder')), {
      target: { value: 'a lighthouse keeper' }
    })
    fireEvent.click(screen.getByText(translateBots('avatar.generate')))

    await expectDeadline()
  })

  it('group picture generation passes IMAGE_GENERATE_TIMEOUT_MS', async () => {
    const { GroupImageControls } = await import('./group-chat-parts')
    render(<GroupImageControls image={null} onImage={vi.fn()} seedMembers={['alpha', 'builder']} seedName="Core" />)

    fireEvent.click(screen.getByText(translateBots('avatar.generate')))

    await expectDeadline()
  })
})
