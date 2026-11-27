import * as fs from 'node:fs'
import * as path from 'node:path'

import { expect, test } from '@playwright/test'

import {
  coreAppEnv,
  createCoreSandbox,
  launchCoreApp,
  recordWebSockets,
  waitForInteractive,
  writeProviderHome
} from './harness'
import { startScriptedProvider } from './provider'

// A group's members do not share the ambient profile's approval policy (#156647).
// Drive real group creation, member execution and denial; only inference is scripted.
test('group approvals stay per profile and never offer the ambient profile mode as a room control', async () => {
  test.setTimeout(240_000)
  const provider = await startScriptedProvider()
  const sandbox = createCoreSandbox('approval-mode')
  const receipt = path.join(sandbox.root, 'command-receipt')
  fs.writeFileSync(receipt, 'synthetic approval probe', { mode: 0o600 })
  writeProviderHome(sandbox.hermesHome, provider.url)
  writeProviderHome(path.join(sandbox.hermesHome, 'profiles', 'programmer'), provider.url, '', 'off')
  writeProviderHome(path.join(sandbox.hermesHome, 'profiles', 'reviewer'), provider.url, '', 'manual')

  const { app, page } = await launchCoreApp(
    coreAppEnv(sandbox, {
      HERMES_INSTALL_STATE_DIR: path.join(sandbox.root, 'install-state'),
      HERMES_DISABLE_LAZY_INSTALLS: '1'
    })
  )

  const wire = recordWebSockets(page)
  const pageErrors: string[] = []
  page.on('pageerror', error => pageErrors.push(String(error)))

  try {
    await waitForInteractive(app, page)
    const statusbar = page.locator('[data-slot="statusbar"]')
    await expect(statusbar.getByRole('button', { name: 'Off', exact: true })).toBeVisible()
    await page
      .getByRole('button', { name: 'Bots', exact: true })
      .or(page.getByRole('tab', { name: 'Bots', exact: true }))
      .first()
      .click()
    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    await page.getByRole('menuitem', { name: 'New Group Chat' }).click()
    const dialog = page.getByRole('dialog', { name: 'New Group Chat' })

    for (const member of ['Programmer', 'Reviewer']) {
      await dialog.getByText(member, { exact: true }).locator('xpath=ancestor::label').getByRole('checkbox').click()
    }

    await dialog.getByRole('textbox', { name: 'Group name' }).fill('Approval probe')
    await dialog.getByRole('button', { name: 'Create group (2)' }).click()
    const composer = page.getByRole('textbox', { name: 'Group message', exact: true }).filter({ visible: true })
    await expect(composer).toBeVisible()
    await expect(statusbar.getByText('Per profile', { exact: true })).toBeVisible()
    await expect(statusbar.getByRole('button', { name: /^(Off|Smart|Manual)$/ })).toHaveCount(0)

    // The keyboard-accessible explanation offers no mode choices.
    await statusbar.getByRole('button', { name: 'Per profile', exact: true }).focus()
    await page.keyboard.press('Enter')
    await expect(
      page.getByText('Approval modes are set per profile. Open a bot or profile to change its mode.')
    ).toBeVisible()
    await expect(page.getByRole('menuitemradio', { name: /^(Off|Smart|Manual)/ })).toHaveCount(0)
    await page.keyboard.press('Escape')

    // Moving focus into the side roster must not retarget the room to the ambient profile.
    await page.getByRole('button', { name: 'New bot or group chat' }).click()
    await page.keyboard.press('Escape')
    await expect(statusbar.getByRole('button', { name: 'Per profile', exact: true })).toBeVisible()
    await expect(statusbar.getByRole('button', { name: /^(Off|Smart|Manual)$/ })).toHaveCount(0)

    for (const [index, member] of ['programmer', 'reviewer'].entries()) {
      const marker = `U${index + 1}-approvalmode`
      const approved = member === 'programmer'
      fs.chmodSync(receipt, 0o600)
      provider.script(marker, [
        { toolCalls: [{ name: 'terminal', args: { command: `chmod 777 ${JSON.stringify(receipt)}` } }] },
        { text: [`DONE-${marker}`] }
      ])
      await composer.fill(`@${member} ${marker} run the synthetic probe exactly once`)
      await composer.press('Enter')
      await expect.poll(() => provider.completions.some(c => c.marker === marker)).toBe(true)

      if (!approved) {
        const deny = page.getByRole('button', { name: 'Deny', exact: true }).filter({ visible: true }).first()
        await expect(deny).toBeVisible()
        await expect(statusbar.getByText('Per profile', { exact: true })).toBeVisible()
        await deny.click()
      }

      await expect
        .poll(() => provider.completions.some(c => c.marker === marker && c.step === 1 && c.finished))
        .toBe(true)
      expect(fs.statSync(receipt).mode & 0o777).toBe(approved ? 0o777 : 0o600)
    }

    const replies = wire.sent.filter(frame => frame.method === 'groups.approve')
    expect(replies).toHaveLength(1)
    expect(replies[0].params).toMatchObject({ member_id: 'reviewer', choice: 'deny' })
    expect(wire.sent.filter(frame => frame.method === 'config.set' && frame.params?.key === 'approvals.mode')).toEqual(
      []
    )

    // The real profile menu comes back on an ordinary session; returning to the room removes it.
    await page.getByRole('tab', { name: 'New session', exact: true }).click()
    await expect(statusbar.getByRole('button', { name: 'Off', exact: true })).toBeVisible()
    await page.getByRole('tab', { name: 'Approval probe', exact: true }).click()
    await expect(statusbar.getByText('Per profile', { exact: true })).toBeVisible()
    await expect(statusbar.getByRole('button', { name: /^(Off|Smart|Manual)$/ })).toHaveCount(0)
    expect(pageErrors).toEqual([])
  } finally {
    await test.info().attach('approval-probe', {
      body: JSON.stringify(
        { requests: wire.sent.filter(frame => /^(groups\.|config\.)/.test(frame.method)), pageErrors },
        null,
        2
      ),
      contentType: 'application/json'
    })
    await app.close().catch(() => undefined)
    await provider.close()
    sandbox.cleanup()
  }
})
