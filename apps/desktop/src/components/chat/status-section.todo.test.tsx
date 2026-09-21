import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import { StatusSection } from './status-section'

describe('StatusSection disclosure accessibility', () => {
  it('links the trigger and body with aria-expanded/aria-controls', () => {
    const html = renderToStaticMarkup(
      <StatusSection defaultCollapsed={false} label="Tasks 1/2">
        <span>Task body</span>
      </StatusSection>
    )

    expect(html).toContain('aria-expanded="true"')
    const control = html.match(/aria-controls="([^"]+)"/)?.[1]
    expect(control).toBeTruthy()
    expect(html).toContain(`id="${control}"`)
    expect(html).toContain('Task body')
  })

  it('supports a controlled collapsed todo disclosure', () => {
    const html = renderToStaticMarkup(
      <StatusSection collapsed label="Tasks 2/2" onCollapsedChange={() => undefined}>
        <span>Finished task</span>
      </StatusSection>
    )

    expect(html).toContain('aria-expanded="false"')
    expect(html).not.toContain('aria-controls=')
    expect(html).not.toContain('Finished task')
  })
})
