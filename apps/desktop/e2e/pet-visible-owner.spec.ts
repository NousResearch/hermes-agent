import fs from 'node:fs'
import path from 'node:path'

import { writeEnvFile, writeMockProviderConfig } from '../../../tests-js/scripts/mock-provider-config'
import { startMockServer } from '../../../tests-js/scripts/mock-server'

import { buildAppEnv, createSandbox, launchDesktop, type MockBackendFixture, waitForAppReady } from './fixtures'
import { RealSessionBuilder } from './real-session-builder'
import { expect, test } from './test'

type Page = MockBackendFixture['page']
interface Receipt {
  method: string
  params: Record<string, unknown>
  endpoint: string
}
interface PetProbe {
  calls: Receipt[]
  states: Record<string, { enabled: boolean; active: string; scale: number }>
}

// Real isolated Electron + serve + bot sessions, with only pet RPC answers
// scripted at the same WebSocket seam as bot-screen-stale-status-reply.spec.ts.
// No petdex/CDN or generation calls: the catalogue and thumbnail are fixtures.
// This proves renderer routing/UI behavior, not Python pet persistence or remote
// connection routing (this rig has exactly one local gateway endpoint).
//
// The launch reading is taken from the RENDERER, not from the backend: with no
// bot row open, the profile the renderer stamps on its own pet.gallery calls
// is the owner it falls back to. Visible owners are then chosen to differ from
// that reading, so a pass can never come from the launch fallback.
//
// Scope: The owner claim covers the pet RPCs sent through the owner wrapper:
// gallery, thumb, info, info.meta, select, disable, scale, rename, remove and
// export (gallery page, settings, floating pet). Two callers do not use the
// wrapper and stay on the launch profile: the generate flow
// (store/pet-generate.ts: pet.generate, pet.hatch and its adopt/discard calls;
// the backend registers generate and hatch scoped=False, methods_session.py:1732
// and :1789, and ignores the profile param) and the Bots avatar picker
// (plugins/hermes-bots/pet.tsx: unscoped pet.thumb and pet.gallery). Each is its
// own card.
async function installPetProbe(page: Page): Promise<void> {
  await page.evaluate(() => {
    const calls: Receipt[] = []
    const states: PetProbe['states'] = {}
    const png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aX1sAAAAASUVORK5CYII='
    const send = WebSocket.prototype.send

    WebSocket.prototype.send = function (this: WebSocket, data: Parameters<WebSocket['send']>[0]) {
      let frame: { id?: number | string; method?: string; params?: Record<string, unknown> }

      try {
        frame = JSON.parse(String(data))
      } catch {
        return send.call(this, data)
      }

      if (!frame.method?.startsWith('pet.')) {
        return send.call(this, data)
      }

      const { method, params = {} } = frame
      // Record the actual profile, including undefined: never silently repair
      // an unscoped request into the owner this test hopes to see.
      calls.push({ method, params, endpoint: this.url.split('?')[0] })
      const profile = String(params.profile ?? 'UNSCOPED')
      const state = (states[profile] ??= { enabled: false, active: '', scale: 0.33 })
      const slug = `${profile}-companion`
      let result: unknown

      switch (method) {
        case 'pet.gallery':
          result = {
            ...state,
            pets: [{ slug, displayName: `${profile} companion`, installed: true, spritesheetUrl: '' }]
          }

          break

        case 'pet.select':
          state.enabled = true
          state.active = String(params.slug)
          result = { ok: true, slug: state.active }

          break

        case 'pet.disable':
          state.enabled = false
          result = { ok: true }

          break

        case 'pet.scale':
          state.scale = Number(params.scale)
          result = { ok: true, scale: state.scale }

          break

        case 'pet.thumb':
          result = { ok: true, dataUri: `data:image/png;base64,${png}` }

          break

        case 'pet.info':

        case 'pet.info.meta':
          result = {
            enabled: state.enabled,
            slug: state.active,
            displayName: `${profile} companion`,
            scale: state.scale,
            spritesheetRevision: `${profile}-fixture`,
            ...(method === 'pet.info'
              ? {
                  mime: 'image/png',
                  spritesheetBase64: png,
                  frameW: 1,
                  frameH: 1,
                  framesPerState: 1,
                  stateRows: ['idle'],
                  framesByState: { idle: 1 }
                }
              : {})
          }

          break

        default:
          // Unknown pet methods must fail closed rather than reaching a CDN.
          queueMicrotask(() =>
            this.dispatchEvent(
              new MessageEvent('message', {
                data: JSON.stringify({
                  jsonrpc: '2.0',
                  id: frame.id,
                  error: { code: -32601, message: `Unscripted ${method}` }
                })
              })
            )
          )

          return
      }

      queueMicrotask(() =>
        this.dispatchEvent(
          new MessageEvent('message', {
            data: JSON.stringify({ jsonrpc: '2.0', id: frame.id, result })
          })
        )
      )
    }

    ;(window as unknown as { __petProbe: PetProbe }).__petProbe = { calls, states }
  })
}

const receipts = (page: Page) => page.evaluate(() => (window as unknown as { __petProbe: PetProbe }).__petProbe.calls)

const BOTS = ['alpha', 'bravo', 'charlie'] as const

async function selectBot(page: Page, owner: string): Promise<void> {
  await page
    .getByRole('button', { name: 'Bots', exact: true })
    .or(page.getByRole('tab', { name: 'Bots', exact: true }))
    .first()
    .click()
  const row = page.locator(`[data-slot="bots-roster"] [data-roster-key="local::${owner}"]`)
  await expect(row).toBeVisible({ timeout: 30_000 })
  await row.click()
  await expect(page.getByText(`Hello ${owner}`, { exact: true }).filter({ visible: true }).first()).toBeVisible({
    timeout: 60_000
  })
}

async function openPetSettings(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Open settings', exact: true }).click()
  await page.getByRole('button', { name: 'Appearance', exact: true }).click()
  await page.getByRole('button', { name: 'Pet', exact: true }).click()
  await expect(page.getByPlaceholder('Search pets…')).toBeVisible()
}

async function closeSettings(page: Page): Promise<void> {
  await page.getByRole('button', { name: 'Close settings', exact: true }).click()
}

// The renderer's own fallback owner: what it stamps on pet.gallery while no
// bot row is open. Several distinct values mean there is no single reading.
async function rendererLaunchReading(page: Page): Promise<string> {
  const start = (await receipts(page)).length
  await openPetSettings(page)
  await expect
    .poll(async () => (await receipts(page)).slice(start).some(call => call.method === 'pet.gallery'))
    .toBe(true)
  await closeSettings(page)

  const profiles = new Set(
    (await receipts(page))
      .slice(start)
      .filter(call => call.method === 'pet.gallery')
      .map(call => String(call.params.profile ?? 'UNSCOPED'))
  )

  expect([...profiles], 'renderer has exactly one launch reading before any bot row opens').toHaveLength(1)

  return [...profiles][0]
}

test('floating pet settings follow visible bot owners B → C → B, all distinct from the renderer launch reading', async () => {
  test.setTimeout(300_000)
  expect(
    fs.existsSync(path.resolve(import.meta.dirname, '../dist/electron-main.mjs')),
    'Build desktop dist before E2E (npm run build in apps/desktop)'
  ).toBe(true)
  const sandbox = createSandbox('pet-visible-owner')
  const mock = await startMockServer()
  let fixture: MockBackendFixture | undefined
  let probeInstalled = false
  const pageErrors: string[] = []
  const seedEnv = process.env

  try {
    writeMockProviderConfig(sandbox.hermesHome, mock.url)
    writeEnvFile(sandbox.hermesHome)
    // RealSessionBuilder inherits process.env; give it the SAME stripped,
    // isolated environment as Electron, not the invoking agent's home/session.
    process.env = buildAppEnv(sandbox)

    try {
      for (const owner of BOTS) {
        const home = path.join(sandbox.hermesHome, 'profiles', owner)
        fs.mkdirSync(home, { recursive: true })
        writeMockProviderConfig(home, mock.url)
        writeEnvFile(home)
        const builder = await RealSessionBuilder.start(home)

        try {
          await builder.createSession({ title: 'Bot Chat', turns: [`Hello ${owner}`] })
        } finally {
          await builder.close()
        }
      }
    } finally {
      process.env = seedEnv
    }

    const { app, page } = await launchDesktop(buildAppEnv(sandbox))
    fixture = {
      app,
      page,
      mock,
      mockUrl: mock.url,
      sandbox,
      cleanup: async () => {
        await app.close()
      }
    }
    page.on('pageerror', error => pageErrors.push(String(error)))
    await waitForAppReady(fixture, 120_000)
    await installPetProbe(page)
    probeInstalled = true

    const launchReading = await rendererLaunchReading(page)
    const owners = BOTS.filter(owner => owner !== launchReading)
    // A pass must name what it proved: the reading and the owners it ran on.
    test.info().annotations.push({ type: 'renderer-launch-reading', description: launchReading })
    // Fail loudly rather than run a transition the launch fallback could pass.
    expect(
      owners.length,
      `need two bot owners distinct from the renderer launch reading "${launchReading}"`
    ).toBeGreaterThanOrEqual(2)
    const [ownerB, ownerC] = owners

    test
      .info()
      .annotations.push(
        { type: 'visible-owner-B', description: ownerB },
        { type: 'visible-owner-C', description: ownerC }
      )

    for (const owner of [ownerB, ownerC, ownerB]) {
      await selectBot(page, owner)
      const start = (await receipts(page)).length
      await openPetSettings(page)
      const card = page.getByRole('button', { name: new RegExp(`${owner} companion`) }).first()
      await expect(card).toBeVisible({ timeout: 15_000 })

      for (const other of new Set([...BOTS, launchReading].filter(name => name !== owner))) {
        await expect(page.getByRole('button', { name: new RegExp(`${other} companion`) })).toHaveCount(0)
      }

      await card.click()
      const toggle = page.getByRole('switch', { name: 'Choose a pet', exact: true })
      await expect(toggle).toBeChecked()
      const slider = page.getByRole('slider', { name: 'Size', exact: true })
      await expect(slider).toBeVisible()
      await slider.focus()
      await slider.press('ArrowRight')
      await expect
        .poll(async () => (await receipts(page)).slice(start).filter(call => call.method === 'pet.scale').length)
        .toBe(1)
      await toggle.click()
      await expect(toggle).not.toBeChecked()
      await toggle.click()
      await expect(toggle).toBeChecked()
      await expect
        .poll(async () => (await receipts(page)).slice(start).filter(call => call.method === 'pet.select').length)
        .toBe(2)

      const calls = (await receipts(page)).slice(start)

      for (const method of ['pet.gallery', 'pet.thumb', 'pet.select', 'pet.scale', 'pet.disable', 'pet.info.meta']) {
        const matching = calls.filter(call => call.method === method)
        expect(matching.length, `${owner}: ${method} traversed the real RPC client`).toBeGreaterThan(0)

        for (const call of matching) {
          expect(call.params.profile, `${method} visible owner`).toBe(owner)
        }
      }

      expect(calls.filter(call => call.method === 'pet.select').map(call => call.params.slug)).toEqual([
        `${owner}-companion`,
        `${owner}-companion`
      ])
      // One endpoint carries every owner, so the owner param is the only router.
      expect(new Set(calls.map(call => call.endpoint)).size).toBe(1)
      await closeSettings(page)
    }

    // OPENS CHECK ONLY. plugins/hermes-bots/pet.tsx is a DIFFERENT gallery:
    // its Pet tab chooses a cropped avatar image, NOT the floating mascot. This
    // step proves the picker opens, previews and cancels without sending mascot
    // mutations. It does NOT prove whose pets the picker lists (it calls
    // pet.gallery unscoped; see the scope note at the top).
    const avatarStart = (await receipts(page)).length
    await page.locator(`[data-slot="bots-roster"] [data-roster-key="local::${ownerB}"]`).click({ button: 'right' })
    await page.getByRole('menuitem', { name: 'Edit…', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Edit profile' })
    await expect(dialog).toBeVisible()
    await dialog.getByRole('button', { name: 'Pet', exact: true }).click()
    await expect(dialog.getByText('Pick a pet as this bot’s profile picture.', { exact: true })).toBeVisible()
    const avatar = dialog.getByRole('button', { name: /companion/ }).first()
    await expect(avatar).toBeVisible()
    await avatar.click()
    // Picking a pet sets a LOCAL preview image; the picker then offers its
    // own remove control (avatar-picker.tsx). Exact name: the Pet tab shows a
    // second, tab-local remove button too.
    await expect(dialog.getByRole('button', { name: 'Remove image — use shape', exact: true })).toBeVisible()
    await dialog.getByRole('button', { name: 'Cancel', exact: true }).click()
    const avatarCalls = (await receipts(page)).slice(avatarStart)
    expect(avatarCalls.some(call => call.method === 'pet.gallery')).toBe(true)
    expect(avatarCalls.filter(call => ['pet.select', 'pet.disable', 'pet.scale'].includes(call.method))).toEqual([])
    expect(pageErrors).toEqual([])
  } finally {
    process.env = seedEnv

    try {
      if (probeInstalled && fixture && !fixture.page.isClosed()) {
        await test.info().attach('pet-wire', {
          body: JSON.stringify(await receipts(fixture.page), null, 2),
          contentType: 'application/json'
        })
      }
    } finally {
      try {
        await fixture?.cleanup()
      } finally {
        await mock.close()
        sandbox.cleanup()
      }
    }
  }
})
