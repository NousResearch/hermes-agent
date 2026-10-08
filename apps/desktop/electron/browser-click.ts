import type { ElementHandle, Page } from 'puppeteer-core'

export async function clickExactElement(page: Page, handle: ElementHandle<Element>, product: string) {
  if (product === 'firefox') { await handle.click(); return }
  // Chromium's click() first waits on IntersectionObserver. Background/occluded
  // windows may not render that callback. Scroll through CDP, then derive the
  // point from this exact handle; never accept caller coordinates or replay.
  await handle.scrollIntoView()
  const point = await handle.clickablePoint()
  const clear = await handle.evaluate((element, point) => {
    const hit = document.elementFromPoint(point.x, point.y)
    return element.isConnected && Boolean(hit && (hit === element || element.contains(hit)))
  }, point)
  if (!clear) throw new Error('The exact referenced element is covered or detached. Read fresh state before acting.')
  await page.mouse.click(point.x, point.y)
}
