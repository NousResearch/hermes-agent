import { translateNow } from '@/i18n'

const RAIL = '[data-tour="profile-rail"]'
const SESSIONS = '[data-tour="sessions-sidebar"]'

async function waitFor(selector: string, timeoutMs = 6000): Promise<boolean> {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const visible = [...document.querySelectorAll(selector)].some(node => {
      const { width, height } = node.getBoundingClientRect()

      return width > 0 && height > 0 && !node.closest('[data-pane-hidden]')
    })

    if (visible) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, 120))
  }

  return false
}

export async function showHandoffTour(): Promise<void> {
  if (!(await waitFor(RAIL))) {
    return
  }

  const sessionsVisible = await waitFor(SESSIONS, 1500)
  const copy = (key: string) => translateNow(`handoffTour.${key}`)
  const { startTour } = await import('@/lib/tour')

  await startTour([
    { accent: true, selector: RAIL, side: 'right', text: copy('profileText'), title: copy('profileTitle') },
    ...(sessionsVisible
      ? [{ selector: SESSIONS, side: 'right' as const, text: copy('sessionsText'), title: copy('sessionsTitle') }]
      : []),
    { accent: true, selector: RAIL, side: 'right', text: copy('stayText'), title: copy('stayTitle') }
  ])
}
