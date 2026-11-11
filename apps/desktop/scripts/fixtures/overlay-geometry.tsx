import { flushSync } from 'react-dom'
import { createRoot } from 'react-dom/client'

import { OverlayMain, OverlaySidebar, OverlaySplitLayout } from '../../src/app/overlays/overlay-split-layout'
import { OverlayView } from '../../src/app/overlays/overlay-view'
import { ModelSelect } from '../../src/app/settings/model-select'
import { ListRow, SettingsContent } from '../../src/app/settings/primitives'
import { Button } from '../../src/components/ui/button'
import { Dialog, DialogContent, DialogTitle } from '../../src/components/ui/dialog'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../src/components/ui/select'
import { Sheet, SheetContent, SheetTitle } from '../../src/components/ui/sheet'
import { SurfaceCloseButton, SurfaceCloseHeader } from '../../src/components/ui/surface-close'

const root = createRoot(document.getElementById('root')!)
const results: object[] = []
const failures: string[] = []

const check = (condition: boolean, message: string) => {
  if (!condition) {
    failures.push(message)
  }
}

const bounds = (el: Element) => el.getBoundingClientRect()

const intersects = (a: DOMRect, b: DOMRect) =>
  a.left < b.right && a.right > b.left && a.top < b.bottom && a.bottom > b.top

const description =
  'هذا نص طويل لاختبار التفاف المحتوى داخل عمود الإعدادات. '.repeat(8) + ' model/' + 'longprovidermodelid'.repeat(20)

function Fields() {
  return (
    <SettingsContent>
      <p data-description>{description}</p>
      {['المزوّد', 'النموذج'].map(label => (
        <ListRow
          action={
            label === 'النموذج' ? (
              <ModelSelect
                aria-label={label}
                className="w-full min-w-0 max-w-full @2xl:w-72"
                models={['fixture-model']}
                onValueChange={() => {}}
                providerSlug="fixture"
                value="fixture-model"
              />
            ) : (
              <Select value="fixture">
                <SelectTrigger aria-label={label} className="w-full min-w-0 max-w-full @2xl:w-72">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="fixture">Fixture value</SelectItem>
                </SelectContent>
              </Select>
            )
          }
          key={label}
          title={label}
        />
      ))}
      <Button>Apply</Button>
    </SettingsContent>
  )
}

async function run() {
  for (const dir of ['rtl', 'ltr']) {
    document.documentElement.dir = dir

    for (const width of [400, 720, 960, 1440]) {
      // Fixed surfaces use the viewport; the CLI opens at this width per invocation.
      if (window.innerWidth !== width) {
        continue
      }

      for (const surface of ['overlay', 'dialog', 'banner', 'sheet', 'compact']) {
        flushSync(() =>
          root.render(
            surface === 'overlay' ? (
              <OverlayView
                edgeBadge={<Button>⌘K بحث</Button>}
                onClose={() => {}}
                titlebarActions={<Button>Action</Button>}
              >
                <OverlaySplitLayout>
                  <OverlaySidebar>
                    <Button>النموذج</Button>
                    <Button>Appearance</Button>
                  </OverlaySidebar>
                  <OverlayMain>
                    <Fields />
                  </OverlayMain>
                </OverlaySplitLayout>
              </OverlayView>
            ) : surface === 'sheet' ? (
              <Sheet open>
                <SheetContent>
                  <SheetTitle>Sheet</SheetTitle>
                  <Fields />
                </SheetContent>
              </Sheet>
            ) : surface === 'compact' ? (
              <div className="w-72">
                <SurfaceCloseHeader close={<SurfaceCloseButton />}>
                  <Button>{'VeryLongActionLabel'.repeat(12)}</Button>
                  <Button>Search</Button>
                </SurfaceCloseHeader>
                <Button>Body</Button>
              </div>
            ) : (
              <Dialog open>
                <DialogContent banner={surface === 'banner' ? 'Notice' : undefined}>
                  <DialogTitle>Settings</DialogTitle>
                  <Fields />
                </DialogContent>
              </Dialog>
            )
          )
        )
        await new Promise(resolve => setTimeout(resolve, 80))
        const close = document.querySelector('[data-slot="surface-close"], [data-slot="dialog-close-button"]')!
        check(Boolean(close), `${surface}/${dir}/${width}: missing close`)

        if (!close) {
          continue
        }

        const box = bounds(close)
        check(box.width >= 24 && box.height >= 24, `${surface}/${dir}/${width}: collapsed close`)
        const header = close.closest('[data-slot="surface-close-header"]')!
        const h = bounds(header)
        check(
          dir === 'rtl' ? box.left < h.left + h.width / 2 : box.right > h.right - h.width / 2,
          `${surface}/${dir}/${width}: wrong corner`
        )

        const controls = [...document.querySelectorAll('button, input, [role="combobox"]')].filter(
          el => el !== close && !close.contains(el) && bounds(el).width > 0
        )

        for (const control of controls) {
          check(
            !intersects(box, bounds(control)),
            `${surface}/${dir}/${width}: close intersects ${control.getAttribute('aria-label') || control.textContent}`
          )
        }

        const fields = [...document.querySelectorAll('[data-slot="select-trigger"]')].map(el => ({
          label: el.getAttribute('aria-label'),
          ...bounds(el).toJSON()
        }))

        const paragraph = document.querySelector('[data-description]')

        if (paragraph) {
          const rect = bounds(paragraph)

          const content =
            paragraph.closest('main') ?? paragraph.closest('[data-slot="dialog-content"], [data-slot="sheet-content"]')!

          const column = bounds(content)
          check(
            rect.left >= column.left && rect.right <= column.right,
            `${surface}/${dir}/${width}: paragraph outside column`
          )
          check(paragraph.scrollWidth <= paragraph.clientWidth + 1, `${surface}/${dir}/${width}: text does not wrap`)

          for (const field of fields) {
            check(field.width >= 100 && field.height >= 24, `${surface}/${dir}/${width}: collapsed ${field.label}`)
            check(
              field.left >= column.left && field.right <= column.right,
              `${surface}/${dir}/${width}: clipped ${field.label}`
            )
          }
        }

        results.push({
          surface,
          dir,
          width,
          close: box.toJSON(),
          fields,
          paragraph: paragraph ? bounds(paragraph).toJSON() : null
        })

        if (window.location.search.includes('preview')) {
          return
        }

        flushSync(() => root.render(null))
        await new Promise(resolve => setTimeout(resolve, 20))
      }
    }
  }

  document.body.innerHTML = ''
  const output = document.createElement('pre')
  output.id = 'geometry-result'
  output.textContent = JSON.stringify({ results, failures })
  document.body.append(output)
}

run().catch(error => {
  document.body.innerHTML = `<pre id="geometry-result">${JSON.stringify({ failures: [String(error)] })}</pre>`
})
