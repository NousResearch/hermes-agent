import { useI18n } from '@/i18n'

// Props are kept for call-site compatibility (the Thread passes the resolved
// personality + seed), but the home screen no longer varies its copy.
export type IntroProps = {
  personality?: string
  seed?: number
}

/**
 * Codex-minimal empty state: a single quiet heading, nothing else — no brand
 * mark, wordmark, tagline, or quick-task grid. The composer below carries the
 * whole interaction, so the launch screen stays as bare as the reference.
 */
export function Intro(_props: IntroProps) {
  const { t } = useI18n()

  return (
    <div
      className="pointer-events-none flex w-full min-w-0 flex-col items-center justify-center px-4 py-6 text-center sm:px-6 lg:px-8"
      data-slot="aui_intro"
    >
      <h1 className="m-0 text-balance text-[1.875rem] font-medium leading-tight tracking-[-0.01em] text-foreground">
        {t.home.title}
      </h1>
    </div>
  )
}
