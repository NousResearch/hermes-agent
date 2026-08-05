import { act, cleanup, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'

import { Dialog, DialogContent } from '@/components/ui/dialog'
import { I18nProvider } from '@/i18n/context'

import { BlockerView } from './updates-overlay'

async function renderWithI18n(ui: React.ReactNode) {
  await act(async () => {
    render(
      <I18nProvider configClient={{ getConfig: async () => ({}), saveConfig: async () => ({ ok: true }) }}>
        <Dialog open>
          <DialogContent>{ui}</DialogContent>
        </Dialog>
      </I18nProvider>
    )
  })
}

describe('BlockerView', () => {
  afterEach(cleanup)

  it('explains safe local previews and offers one-click close-and-update', async () => {
    const onStopAndUpdate = vi.fn()

    await renderWithI18n(
      <BlockerView
        blockers={[
          {
            pid: 47484,
            name: 'python.exe',
            cmdline: 'python.exe -m http.server 8766',
            kind: 'local-preview',
            safeToStop: true,
            label: 'Example Preview',
            port: 8766
          }
        ]}
        onDismiss={() => {}}
        onStopAndUpdate={onStopAndUpdate}
      />
    )

    expect(screen.getByText('Close local previews to update Hermes?')).toBeTruthy()
    expect(screen.getByText('Example Preview')).toBeTruthy()
    expect(screen.getByText('Port 8766')).toBeTruthy()
    expect(screen.getByText(/will not modify or delete your files/i)).toBeTruthy()

    fireEvent.click(screen.getByRole('button', { name: 'Close previews and update' }))
    expect(onStopAndUpdate).toHaveBeenCalledTimes(1)
  })
})
