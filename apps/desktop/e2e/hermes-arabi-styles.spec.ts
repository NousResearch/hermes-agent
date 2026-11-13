import { mkdtempSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import * as path from 'node:path'

import type { Page } from '@playwright/test'
import { createServer } from 'vite'

import { resolveElectronBinary } from './electron-binary'
import { _electron, expect, test } from './test'

// Synthetic CSS fixture: production data-slot contracts and utilities, no gateway,
// user sessions or credentials. Its Electron process has its own disposable data.
const fixture = `
<main style="max-width:760px;margin:40px auto;padding:24px">
  <div data-slot="aui_thread-content" class="flex flex-col gap-4">
    <div data-slot="aui_user-message-root" class="flex flex-col items-stretch">
      <div data-slot="aui_user-bubble-actions" class="relative w-full max-w-full">
      <div class="human-message-with-todos-wrapper flex w-full flex-col gap-0"><div class="relative w-full">
      <div class="composer-human-message standalone-glass relative flex w-full min-w-0 max-w-full flex-col gap-1.5 rounded-xl border bg-(--dt-user-bubble) px-3 py-2 pr-9 text-left">
        <div data-slot="aui_user-inline-text" id="user-text">هذه رسالة عربية تجريبية لقياس وضوح الخط ووزنه العادي.</div>
      </div>
      </div></div></div>
    </div>
    <div data-slot="aui_assistant-message-root" class="flex w-full min-w-0 flex-col self-start">
      <div data-slot="aui_assistant-message-content" class="min-w-0 max-w-full text-foreground">
        <div class="aui-md prose"><p id="arabic-prose">هذا رد تجريبي بالعربية. النص العادي واضح وخفيف، مع <strong>تأكيد مقصود</strong> يبقى بارزًا.</p><p>نص مختلط Hermes 123 مع العربية.</p></div>
      </div>
    </div>
  </div>
  <div data-slot="composer-dock" style="margin-block-start:48px">
    <div data-slot="composer-root">
      <div data-slot="composer-surface" class="relative rounded-2xl border bg-(--composer-fill) p-3">
        <div data-slot="composer-rich-input" id="editor" contenteditable="true" role="textbox" aria-label="Synthetic prompt" style="min-height:64px;outline:none">اكتب رسالتك هنا لاختبار التحديد ومؤشر الكتابة.</div>
        <button type="button">إرسال</button>
      </div>
    </div>
  </div>
</main>`

// The real ThemeProvider writes palette seeds as well as the dark class. A
// class-only mock leaves light foreground seeds in a dark screenshot. Use the
// audited synthetic skin palette, never a signed-in window or user skin file.
async function setFixtureMode(page: Page, mode: 'light' | 'dark') {
  await page.evaluate(mode => {
    const root = document.documentElement
    root.classList.toggle('dark', mode === 'dark')
    root.dataset.hermesTheme = 'hermes-arabi'
    root.dataset.hermesMode = mode
    root.style.colorScheme = mode
    root.style.setProperty('--theme-foreground', mode === 'dark' ? '#f5f5f5' : '#1c1c1c')
    root.style.setProperty('--theme-background-seed', mode === 'dark' ? '#121212' : '#fcfcfc')
    root.style.setProperty('--theme-card-seed', mode === 'dark' ? '#161616' : '#ffffff')
    root.style.setProperty('--theme-neutral-card', mode === 'dark' ? '#161616' : '#ffffff')
    root.style.setProperty('--theme-mix-chrome', '100%')
    ;(document.activeElement as HTMLElement)?.blur()
  }, mode)
}

async function openFixture() {
  const temporary = mkdtempSync(path.join(tmpdir(), 'hermes-arabi-styles-'))
  const main = path.join(temporary, 'main.cjs')
  writeFileSync(
    main,
    `const { app, BrowserWindow } = require('electron');
app.setPath('userData', ${JSON.stringify(path.join(temporary, 'user-data'))});
app.whenReady().then(() => {
  const window = new BrowserWindow({ width: 900, height: 650,
    webPreferences: { sandbox: true, contextIsolation: true } });
  window.loadURL('about:blank');
});`
  )
  const server = await createServer({ server: { host: '127.0.0.1', port: 0 } })
  await server.listen()

  const app = await _electron.launch({
    executablePath: resolveElectronBinary([path.resolve('.'), path.resolve('../..')]),
    args: [main]
  })

  const page = await app.firstWindow()
  await page.setContent(fixture)
  await page.addStyleTag({ url: `${server.resolvedUrls!.local[0]}src/styles.css?direct` })
  await page.evaluate(() => {
    document.documentElement.lang = 'ar'
    document.documentElement.dir = 'rtl'
  })

  await setFixtureMode(page, 'light')

  return { app, page, server }
}

test('Arabic chat and editor use native regular typography without flattening emphasis', async ({
  browserName: _browserName
}, testInfo) => {
  const { app, page, server } = await openFixture()

  try {
    for (const mode of ['light', 'dark'] as const) {
      await setFixtureMode(page, mode)

      const styles = await page.locator('#arabic-prose').evaluate(el => {
        const s = getComputedStyle(el)

        return { family: s.fontFamily, weight: s.fontWeight }
      })

      const cdp = await page.context().newCDPSession(page)
      await cdp.send('DOM.enable')
      await cdp.send('CSS.enable')
      const { root } = await cdp.send('DOM.getDocument')
      const { nodeId } = await cdp.send('DOM.querySelector', { nodeId: root.nodeId, selector: '#arabic-prose' })
      const fonts = await cdp.send('CSS.getPlatformFontsForNode', { nodeId })
      const fontEvidence = JSON.stringify({ styles, fonts }, null, 2)
      writeFileSync(testInfo.outputPath(`font-${mode}.json`), fontEvidence)
      await testInfo.attach(`font-${mode}.json`, { body: fontEvidence, contentType: 'application/json' })
      await cdp.detach()
      await page.screenshot({
        path: testInfo.outputPath(`font-${process.env.HERMES_STYLE_CAPTURE_LABEL ?? 'after'}-${mode}.png`)
      })
      expect.soft(styles.family).toMatch(/^system-ui|^"SF Arabic"/)
      expect.soft(styles.weight).toBe('400')
      expect(fonts.fonts.some(font => font.familyName.includes('Segoe'))).toBe(false)
      await expect(page.locator('#arabic-prose')).toHaveCSS('font-size', '14px')
      await expect(page.locator('#arabic-prose')).toHaveCSS('line-height', '22px')
      await expect(page.locator('#user-text')).toHaveCSS('font-weight', '400')
      await expect(page.locator('#editor')).toHaveCSS('font-weight', '400')
      expect(
        Number(await page.locator('#arabic-prose strong').evaluate(el => getComputedStyle(el).fontWeight))
      ).toBeGreaterThan(400)
    }

    // Language scoping must leave English and the existing CJK editor rules alone.
    await page.locator('#editor').evaluate(el => {
      el.setAttribute('lang', 'ja')
    })
    await expect(page.locator('#editor')).toHaveCSS('font-family', /Hiragino Sans/)
  } finally {
    await app.close()
    await server.close()
  }
})

test('Arabic skins own distinct tokenized bubbles with the user on the right', async ({
  browserName: _browserName
}, testInfo) => {
  const { app, page, server } = await openFixture()

  try {
    const user = page.locator('.composer-human-message')
    const assistant = page.locator('[data-slot="aui_assistant-message-content"]')
    await page.locator('#user-text').evaluate(el => {
      el.textContent = 'نص تجريبي طويل لاختبار عرض الفقاعة '.repeat(8)
    })
    await page.locator('#arabic-prose').evaluate(el => {
      el.textContent = 'رد تجريبي طويل لاختبار الحد الأقصى للعرض '.repeat(8)
    })

    const dialogFill = await page.evaluate(() =>
      getComputedStyle(document.documentElement).getPropertyValue('--ui-chat-bubble-background')
    )

    for (const mode of ['light', 'dark'] as const) {
      await setFixtureMode(page, mode)

      for (const direction of ['rtl', 'ltr'] as const) {
        await page.evaluate(direction => {
          document.documentElement.dir = direction
        }, direction)
        await page.screenshot({ path: testInfo.outputPath(`bubbles-${mode}-${direction}.png`) })
        await expect(user).toHaveCSS('border-top-left-radius', '18px')
        await expect.soft(assistant).toHaveCSS('border-bottom-right-radius', '18px')
        await expect.soft(user).toHaveCSS('padding-top', '7px')
        await expect.soft(assistant).toHaveCSS('padding-bottom', '7px')
        await expect
          .soft(user)
          .toHaveCSS('background-color', mode === 'light' ? 'rgb(28, 28, 28)' : 'rgba(245, 245, 245, 0.18)')
        await expect.soft(user).toHaveCSS('color', mode === 'light' ? 'rgb(255, 255, 255)' : 'rgb(245, 245, 245)')
        await expect
          .soft(assistant)
          .toHaveCSS('background-color', mode === 'light' ? 'rgba(28, 28, 28, 0.06)' : 'rgba(245, 245, 245, 0.04)')

        const bounds = await page.locator('[data-slot="aui_thread-content"]').evaluate(el => {
          const rect = el.getBoundingClientRect()
          const style = getComputedStyle(el)
          const left = Number.parseFloat(style.paddingLeft)
          const right = Number.parseFloat(style.paddingRight)

          return { x: rect.x + left, width: rect.width - left - right }
        })

        const u = (await user.boundingBox())!
        const a = (await assistant.boundingBox())!
        const cap = Math.min(bounds.width * 0.8, 560, bounds.width - 82)
        expect.soft(u.width).toBeLessThanOrEqual(cap + 0.5)
        expect.soft(u.width).toBeCloseTo(cap, 0)
        expect.soft(a.width).toBeLessThanOrEqual(cap + 0.5)
        // This skin keeps physical sides in both directions — user on the
        // right, assistant on the left (RTL: user = inline-start).
        expect.soft(Math.abs(u.x + u.width - (bounds!.x + bounds!.width))).toBeLessThan(1)
        expect.soft(Math.abs(a.x - bounds!.x)).toBeLessThan(1)
      }
    }

    await page.evaluate(() => {
      document.documentElement.dataset.hermesTheme = 'hermes-arabi-light'
      document.documentElement.dataset.hermesMode = 'light'
    })
    await expect(user).toHaveCSS('background-color', 'rgb(28, 28, 28)')
    // The skin's existing customCSS channel can override dedicated tokens only.
    await page.addStyleTag({
      content: ':root { --dt-chat-user-background: rgb(12, 34, 56); --dt-chat-bubble-radius: 23px; }'
    })
    await expect(user).toHaveCSS('background-color', 'rgb(12, 34, 56)')
    await expect(assistant).toHaveCSS('border-top-left-radius', '23px')
    await page.setViewportSize({ width: 320, height: 650 })
    expect((await user.boundingBox())!.width).toBeLessThanOrEqual(272)
    expect((await assistant.boundingBox())!.width).toBeLessThanOrEqual(272)
    await setFixtureMode(page, 'light')
    await page.evaluate(() => {
      document.documentElement.dataset.hermesTheme = 'nous'
    })
    await expect(assistant).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)')
    expect(
      await page.evaluate(() =>
        getComputedStyle(document.documentElement).getPropertyValue('--ui-chat-bubble-background')
      )
    ).toBe(dialogFill)
  } finally {
    await app.close()
    await server.close()
  }
})

test('Arabic composer matches Grok rest, hover, focus, selection and caret in both modes', async ({
  browserName: _browserName
}, testInfo) => {
  const { app, page, server } = await openFixture()

  try {
    const surface = page.locator('[data-slot="composer-surface"]')
    const editor = page.locator('#editor')

    for (const mode of ['light', 'dark'] as const) {
      await setFixtureMode(page, mode)
      await page.mouse.move(0, 0)
      await page.screenshot({ path: testInfo.outputPath(`composer-${mode}-rest.png`) })
      await expect(surface).toHaveCSS('background-color', mode === 'light' ? 'rgb(255, 255, 255)' : 'rgb(31, 31, 31)')
      await expect(surface).toHaveCSS(
        'border-color',
        mode === 'light' ? 'rgba(28, 28, 28, 0.12)' : 'rgba(245, 245, 245, 0.1)'
      )
      await expect(surface).toHaveCSS('border-radius', '22px')
      await page.locator('[data-slot="composer-root"]').evaluate(el => el.setAttribute('data-composer-expanded', ''))
      await expect(surface).toHaveCSS('border-radius', '18px')
      await page.locator('[data-slot="composer-root"]').evaluate(el => el.removeAttribute('data-composer-expanded'))
      await surface.hover()
      const focused = mode === 'light' ? 'rgba(28, 28, 28, 0.24)' : 'rgba(245, 245, 245, 0.22)'
      await expect(surface).toHaveCSS('border-color', focused)
      await editor.click()
      await page.mouse.move(0, 0)
      await expect(surface).toHaveCSS('border-color', focused)
      await expect(editor).toHaveCSS('background-color', 'rgba(0, 0, 0, 0)')
      await expect(editor).toHaveCSS('caret-color', mode === 'light' ? 'rgb(28, 28, 28)' : 'rgb(245, 245, 245)')
      const shadow = await surface.evaluate(el => getComputedStyle(el).boxShadow)
      expect(shadow).toContain('0px 2px 8px -1px')
      expect(shadow).toContain('rgba(0, 0, 0, 0.05)')
      await page.screenshot({ path: testInfo.outputPath(`composer-${mode}-focus.png`), caret: 'initial' })
      await editor.evaluate(el => {
        const selection = window.getSelection()!
        selection.selectAllChildren(el)
      })

      const selectionStyle = await editor.evaluate(el => ({
        background: getComputedStyle(el, '::selection').backgroundColor,
        color: getComputedStyle(el, '::selection').color
      }))

      expect(selectionStyle.background).toBe(mode === 'light' ? 'rgba(28, 28, 28, 0.08)' : 'rgba(245, 245, 245, 0.08)')
      expect(selectionStyle.color).toBe(mode === 'light' ? 'rgb(28, 28, 28)' : 'rgb(245, 245, 245)')
      await page.screenshot({ path: testInfo.outputPath(`composer-${mode}-selection.png`) })
      await page.locator('[data-slot="composer-root"]').evaluate(el => {
        el.setAttribute('data-thread-scrolled-up', '')
      })
      await expect(surface).toHaveCSS('background-color', mode === 'light' ? 'rgb(255, 255, 255)' : 'rgb(31, 31, 31)')
      await page.locator('button').focus()
      await expect(surface).toHaveCSS('border-color', focused)
      await page.keyboard.press('Tab')
      await page.evaluate(() => {
        ;(document.activeElement as HTMLElement)?.blur()
        window.getSelection()?.removeAllRanges()
      })
      await expect(surface).toHaveCSS(
        'border-color',
        mode === 'light' ? 'rgba(28, 28, 28, 0.12)' : 'rgba(245, 245, 245, 0.1)'
      )
    }

    await page.addStyleTag({
      content:
        ':root { --dt-chat-composer-background: rgb(22, 33, 44); --dt-chat-composer-border-focus: rgb(12, 34, 56); }'
    })
    await editor.focus()
    await expect(surface).toHaveCSS('background-color', 'rgb(22, 33, 44)')
    await expect(surface).toHaveCSS('border-color', 'rgb(12, 34, 56)')
  } finally {
    await app.close()
    await server.close()
  }
})
