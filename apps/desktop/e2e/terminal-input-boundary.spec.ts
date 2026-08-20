import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'
import { expect, test } from './test'

interface TerminalDesktopBridge {
  openSessionInTerminal: (
    sessionId: string,
    opts?: { cwd?: string; profile?: string }
  ) => Promise<{
    ok: boolean
    error?: string
  }>
}

test('the external-terminal IPC rejects control-character inputs before launcher creation', async () => {
  let fixture: MockBackendFixture | null = null

  try {
    fixture = await setupMockBackend()
    await waitForAppReady(fixture)

    const results = await fixture.page.evaluate(() => {
      const desktop = (window as unknown as { hermesDesktop: TerminalDesktopBridge }).hermesDesktop

      return Promise.all([
        desktop.openSessionInTerminal('session\r\nstart calc.exe', { profile: 'default' }),
        desktop.openSessionInTerminal('session', { profile: 'default\r\nstart calc.exe' })
      ])
    })

    expect(results).toEqual([
      { ok: false, error: 'invalid-session-id' },
      { ok: false, error: 'invalid-profile' }
    ])
  } finally {
    await fixture?.cleanup()
  }
})
