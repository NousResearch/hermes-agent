import { requestComposerInsert } from '@/app/chat/composer/focus'
import { Codicon } from '@/components/ui/codicon'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { cn } from '@/lib/utils'

const assetPath = (path: string) => `${import.meta.env.BASE_URL}${path.replace(/^\/+/, '')}`

// The ApexNodes app icon doubles as the home-screen brand mark — small product
// chrome, not a splash wordmark (see APEXNODES-UI-REDESIGN §9.5).
export const HOME_BRAND_ICON = 'apple-touch-icon.png'

// Props are kept for call-site compatibility (the Thread passes the resolved
// personality + seed), but the de-developer-ified home screen no longer varies
// its copy by personality — the ApexNodes pitch is fixed.
export type IntroProps = {
  personality?: string
  seed?: number
}

/**
 * Drop a quick-task's prompt into the main composer and focus it. We insert
 * rather than auto-send so the user can tweak before sending — and so the home
 * screen never reaches into the message-send / gateway path. Same external
 * composer bus the sidebar / file-drop already use.
 */
export function sendQuickTask(prompt: string): void {
  triggerHaptic('selection')
  requestComposerInsert(prompt, { mode: 'block', target: 'main' })
}

function QuickTaskCard({ icon, label, prompt }: { icon: string; label: string; prompt: string }) {
  return (
    <button
      className={cn(
        'pointer-events-auto group flex w-full items-center gap-3 rounded-[0.875rem] border border-(--ui-stroke-tertiary) bg-(--ui-bg-editor) px-3.5 py-2.5 text-left',
        'transition-colors duration-100',
        'hover:border-(--ui-stroke-secondary) hover:bg-(--ui-row-hover-background)',
        'focus-visible:border-[color-mix(in_srgb,var(--ui-accent)_45%,var(--ui-stroke-secondary))] focus-visible:outline-none'
      )}
      onClick={() => sendQuickTask(prompt)}
      type="button"
    >
      {/* Neutral by default; the faint violet accent only warms in on hover —
          keeps the home grid gray-quiet like the reference. */}
      <span
        aria-hidden
        className="grid size-8 shrink-0 place-items-center rounded-[0.625rem] bg-(--ui-bg-tertiary) text-(--ui-text-tertiary) transition-colors group-hover:bg-[color-mix(in_srgb,var(--ui-accent)_12%,transparent)] group-hover:text-(--ui-accent)"
      >
        <Codicon name={icon} size="1rem" />
      </span>
      <span className="min-w-0 flex-1 truncate text-[length:var(--conversation-text-font-size)] font-medium text-(--ui-text-secondary) group-hover:text-foreground">
        {label}
      </span>
      <Codicon
        aria-hidden
        className="shrink-0 text-(--ui-text-quaternary) transition-colors group-hover:text-(--ui-accent)"
        name="arrow-right"
        size="0.875rem"
      />
    </button>
  )
}

export function Intro(_props: IntroProps) {
  const { t } = useI18n()
  const home = t.home

  return (
    <div
      className="pointer-events-none flex w-full min-w-0 flex-col items-center justify-center px-4 py-6 text-center sm:px-6 lg:px-8"
      data-slot="aui_intro"
    >
      <div className="flex w-full max-w-[34rem] min-w-0 flex-col items-center">
        <img
          alt=""
          aria-hidden
          className="mb-3.5 size-11 rounded-[0.875rem]"
          src={assetPath(HOME_BRAND_ICON)}
        />

        <div className="mb-2.5 flex items-center gap-1.5 text-[0.6875rem] font-medium tracking-[0.03em] text-(--ui-text-quaternary)">
          <span>{home.wordmark}</span>
          <span aria-hidden className="opacity-60">
            ·
          </span>
          <span>{home.engineBacking}</span>
        </div>

        {/* Claude-style hero: a single quiet question, lighter weight + larger
            than the old bold title, near-black per the reference. */}
        <h1 className="m-0 text-balance text-[1.875rem] font-medium leading-tight tracking-[-0.01em] text-foreground">
          {home.title}
        </h1>

        <p className="mt-2.5 mb-7 max-w-[28rem] text-pretty text-[0.875rem] leading-relaxed text-(--ui-text-tertiary)">
          {home.subtitle}
        </p>

        <div className="w-full">
          <p className="mb-2 self-start text-left text-[0.75rem] font-medium text-(--ui-text-quaternary)">
            {home.quickTasksLabel}
          </p>
          <div className="grid w-full grid-cols-1 gap-2 sm:grid-cols-2">
            {home.quickTasks.map(task => (
              <QuickTaskCard icon={task.icon} key={task.id} label={task.label} prompt={task.prompt} />
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
