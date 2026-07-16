import { useStore } from '@nanostores/react'
import type { ReactNode } from 'react'

import { KbdCombo } from '@/components/ui/kbd'
import { useI18n } from '@/i18n'
import { $composerEnterSends } from '@/store/composer-prefs'

import { COMPLETION_DRAWER_CLASS } from './completion-drawer'

const COMMON_COMMAND_KEYS = ['/help', '/clear', '/resume', '/details', '/copy', '/quit']

/** Stable ids → i18n `hotkeyDescs` keys. Combos resolve mod labels per OS. */
const DEFAULT_COMPOSER_HOTKEY_ROWS = [
  { id: 'composer.mention', combos: ['@'] },
  { id: 'composer.slash', combos: ['/'] },
  { id: 'composer.help', combos: ['?'] },
  { id: 'composer.sendNewline', combos: ['enter', 'shift+enter'] },
  { id: 'composer.sendQueued', combos: ['mod+shift+k'] },
  { id: 'keybinds.openPanel', combos: ['mod+/'] },
  { id: 'composer.cancel', combos: ['escape'] },
  { id: 'composer.history', combos: ['up', 'down'] }
] as const

const MULTILINE_COMPOSER_HOTKEY_ROWS = [
  { id: 'composer.mention', combos: ['@'] },
  { id: 'composer.slash', combos: ['/'] },
  { id: 'composer.help', combos: ['?'] },
  { id: 'composer.newline', combos: ['enter'] },
  { id: 'composer.send', combos: ['mod+enter'] },
  { id: 'composer.steer', combos: ['shift+enter'] },
  { id: 'composer.sendQueued', combos: ['mod+shift+k'] },
  { id: 'keybinds.openPanel', combos: ['mod+/'] },
  { id: 'composer.cancel', combos: ['escape'] },
  { id: 'composer.history', combos: ['up', 'down'] }
] as const

export function HelpHint() {
  const { t } = useI18n()
  const enterSends = useStore($composerEnterSends)
  const c = t.composer
  const hotkeyRows = enterSends ? DEFAULT_COMPOSER_HOTKEY_ROWS : MULTILINE_COMPOSER_HOTKEY_ROWS

  return (
    <div className={COMPLETION_DRAWER_CLASS} data-slot="composer-completion-drawer" data-state="open" role="dialog">
      <Section title={c.commonCommands}>
        {COMMON_COMMAND_KEYS.map(key => (
          <Row description={c.commandDescs[key] ?? ''} key={key} keyLabel={key} mono />
        ))}
      </Section>

      <Section title={c.hotkeys}>
        {hotkeyRows.map(row => (
          <HotkeyRow
            combos={[...row.combos]}
            description={c.hotkeyDescs[row.id] ?? t.keybinds.actions[row.id] ?? ''}
            key={row.id}
          />
        ))}
      </Section>

      <p className="px-2.5 py-1 text-xs text-muted-foreground/80">
        <span className="font-mono text-foreground/80">/help</span> {c.helpFooter}
      </p>
    </div>
  )
}

function Section({ children, title }: { children: ReactNode; title: string }) {
  return (
    <div className="grid gap-0.5 pt-0.5">
      <p className="px-2.5 pb-0.5 pt-1 text-[0.65rem] font-medium uppercase tracking-wide text-muted-foreground/75">
        {title}
      </p>
      {children}
    </div>
  )
}

function Row({ description, keyLabel, mono = false }: { description: string; keyLabel: string; mono?: boolean }) {
  return (
    <div className="flex min-w-0 items-baseline gap-2 rounded-md px-2.5 py-1 text-xs">
      <span
        className={
          mono ? 'shrink-0 truncate font-mono font-medium text-foreground/85' : 'shrink-0 truncate text-foreground/85'
        }
      >
        {keyLabel}
      </span>
      <span className="min-w-0 truncate text-muted-foreground/80">{description}</span>
    </div>
  )
}

function HotkeyRow({ combos, description }: { combos: string[]; description: string }) {
  return (
    <div className="flex min-w-0 items-center gap-2 rounded-md px-2.5 py-1 text-xs">
      <span className="flex shrink-0 items-center gap-1">
        {combos.map(combo => (
          <KbdCombo combo={combo} key={combo} size="sm" />
        ))}
      </span>
      <span className="min-w-0 truncate text-muted-foreground/80">{description}</span>
    </div>
  )
}
