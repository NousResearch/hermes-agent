import { expect, type Locator, type Page, test } from '@playwright/test'

import { type MockBackendFixture, setupMockBackend, waitForAppReady } from './fixtures'

interface SidebarLayout {
  contentOverflowY: string
  sectionBodyOverflows: string[]
  navBottom: number
  profileTop: number
  sectionsBottom: number
  sectionsOverflowY: string
  sectionsTop: number
  sidebarBottom: number
  sidebarTop: number
}

async function scrollMetrics(
  target: Locator
): Promise<{ clientHeight: number; scrollHeight: number; scrollTop: number }> {
  return target.first().evaluate(element => {
    let owner: HTMLElement | null = element as HTMLElement

    while (owner && !['auto', 'scroll'].includes(getComputedStyle(owner).overflowY)) {
      owner = owner.parentElement
    }

    if (!owner) {
      throw new Error('No vertical scroll owner found')
    }

    return { clientHeight: owner.clientHeight, scrollHeight: owner.scrollHeight, scrollTop: owner.scrollTop }
  })
}

async function resetScroll(target: Locator): Promise<void> {
  await target.first().evaluate(element => {
    let owner: HTMLElement | null = element as HTMLElement

    while (owner && !['auto', 'scroll'].includes(getComputedStyle(owner).overflowY)) {
      owner = owner.parentElement
    }

    owner?.scrollTo({ top: 0 })
  })
}

async function sidebarLayout(page: Page): Promise<SidebarLayout> {
  return page.evaluate(() => {
    const sections = document.querySelector<HTMLElement>('[data-sidebar-sections]')!
    const sidebar = sections.closest<HTMLElement>('[data-slot="sidebar"]')!
    const content = sections.closest<HTMLElement>('[data-sidebar="content"]')!
    const nav = content.firstElementChild as HTMLElement
    const profile = content.lastElementChild as HTMLElement
    const bodies = [...sections.querySelectorAll<HTMLElement>('[data-sidebar="group-content"]')]
    const sidebarRect = sidebar.getBoundingClientRect()
    const sectionsRect = sections.getBoundingClientRect()

    return {
      contentOverflowY: getComputedStyle(content).overflowY,
      navBottom: nav.getBoundingClientRect().bottom,
      profileTop: profile.getBoundingClientRect().top,
      sectionBodyOverflows: bodies.map(body => getComputedStyle(body).overflowY),
      sectionsBottom: sectionsRect.bottom,
      sectionsOverflowY: getComputedStyle(sections).overflowY,
      sectionsTop: sectionsRect.top,
      sidebarBottom: sidebarRect.bottom,
      sidebarTop: sidebarRect.top
    }
  })
}

function expectAnchoredLayout(layout: SidebarLayout): void {
  expect(layout.contentOverflowY).toBe('hidden')
  expect(layout.sectionsOverflowY).toBe('auto')
  expect(layout.sectionBodyOverflows.length).toBeGreaterThan(0)
  expect(layout.sectionBodyOverflows.every(overflow => overflow === 'visible')).toBe(true)
  expect(layout.navBottom).toBeLessThanOrEqual(layout.sectionsTop)
  expect(layout.sectionsBottom).toBeLessThanOrEqual(layout.profileTop)
  expect(layout.profileTop).toBeGreaterThanOrEqual(layout.sidebarTop)
  expect(layout.profileTop).toBeLessThan(layout.sidebarBottom)
}

test.describe('sidebar wheel scrolling', () => {
  test.setTimeout(180_000)

  let fixture: MockBackendFixture

  test.beforeAll(async () => {
    fixture = await setupMockBackend()
    await waitForAppReady(fixture, 120_000)
  })

  test.afterAll(async () => {
    await fixture?.cleanup()
  })

  test('anchors shell chrome while the section stack scrolls from headers and rows', async () => {
    const { app, page } = fixture

    const composer = page.locator('[contenteditable="true"]').first()
    await composer.click()
    await composer.fill('Create a sidebar session for the layout check')
    await page.keyboard.press('Enter')
    await page.locator('[data-sidebar-sections]').waitFor({ state: 'visible', timeout: 30_000 })

    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setSize(1220, 800))
    await page.waitForTimeout(250)
    expectAnchoredLayout(await sidebarLayout(page))

    await app.evaluate(({ BrowserWindow }) => BrowserWindow.getAllWindows()[0]?.setSize(1000, 560))
    await page.waitForTimeout(250)
    expectAnchoredLayout(await sidebarLayout(page))

    await page.locator('[data-sidebar-sections]').evaluate(sections => {
      const filler = document.createElement('div')
      filler.dataset.scrollFixture = ''
      filler.style.flex = '0 0 800px'
      sections.append(filler)
    })

    const row = page.locator('[data-sidebar-sections] [data-slot="row-button"]').last()
    const header = page.locator('[data-sidebar-sections] button').first()
    const beforeHeader = await scrollMetrics(header)

    expect(beforeHeader.scrollHeight).toBeGreaterThan(beforeHeader.clientHeight)
    await header.hover()
    await page.mouse.wheel(0, 500)
    await expect.poll(async () => (await scrollMetrics(header)).scrollTop).toBeGreaterThan(beforeHeader.scrollTop)

    await resetScroll(header)
    await row.hover()
    await page.mouse.wheel(0, 500)
    await expect.poll(async () => (await scrollMetrics(row)).scrollTop).toBeGreaterThan(0)
  })
})
