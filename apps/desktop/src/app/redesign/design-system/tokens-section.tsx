import { useStore } from '@nanostores/react'
import { type CSSProperties, type ReactNode, type RefObject, useEffect, useRef, useState } from 'react'

import { CopyButton } from '@/components/ui/copy-button'
import { type DesignToken, hexColor, type SpecimenKind, specimenKind, type TokenFamily } from '@/lib/design-tokens'
import { $portalAccent } from '@/store/portal-accent'
import { useTheme } from '@/themes/context'

import { $widthClass } from '../breakpoints'

import type { DesignSystemCopy } from './design-system-copy'

/** Non-language samples: a type specimen is letterforms, not copy. */
const LETTERS = 'Ag'
const CAPS = 'ABC 123'

const supports = (property: string, value: string) => CSS.supports(property, value)

const SPECIMENS: Record<SpecimenKind, () => ReactNode> = {
  color: () => <span className="ds-swatch" data-swatch="" />,
  duration: () => <MotionSample />,
  easing: () => <MotionSample />,
  font: () => <span className="ds-font-sample">{LETTERS}</span>,
  layer: () => null,
  length: () => <span className="ds-length-bar" />,
  radius: () => <span className="ds-radius-sample" />,
  shadow: () => <span className="ds-shadow-sample" />,
  tracking: () => <span className="ds-tracking-sample">{CAPS}</span>,
  'type-size': () => <span className="ds-type-sample">{LETTERS}</span>,
  value: () => null
}

function MotionSample() {
  return (
    <span className="ds-motion-track">
      <span className="ds-motion-dot" />
    </span>
  )
}

interface Resolved {
  value: string
  kind: SpecimenKind
}

/** The token's value where it is used (this page, in the current theme, accent and width class) and
 *  how to show it. Read a frame late: the theme applies its classes in an effect of its own. */
function useResolvedToken(ref: RefObject<HTMLElement | null>, name: string, revision: string): Resolved | null {
  const [resolved, setResolved] = useState<Resolved | null>(null)

  useEffect(() => {
    const frame = requestAnimationFrame(() => {
      const element = ref.current
      const value = element ? getComputedStyle(element).getPropertyValue(name).replace(/\s+/g, ' ').trim() : ''

      setResolved(value ? { value, kind: specimenKind(name, value, supports) } : null)
    })

    return () => cancelAnimationFrame(frame)
  }, [name, ref, revision])

  return resolved
}

/** A colour token's value as the swatch paints it: color-mix and var() resolved to one colour. */
function useSwatchColor(ref: RefObject<HTMLElement | null>, resolved: Resolved | null, revision: string): string {
  const [color, setColor] = useState('')

  useEffect(() => {
    const swatch = resolved?.kind === 'color' ? ref.current?.querySelector('[data-swatch]') : null

    setColor(swatch ? hexColor(getComputedStyle(swatch).backgroundColor) : '')
  }, [ref, resolved, revision])

  return color
}

/** The source definition: the base value, and the light one where it differs. */
function definedAs(token: DesignToken, copy: DesignSystemCopy): string[] {
  const { base, dark, light } = token.values
  const shared = dark ?? base

  const lines =
    light && light !== shared ? [`${copy.dark}: ${shared ?? '—'}`, `${copy.light}: ${light}`] : [shared ?? light ?? '']

  return [...lines, ...token.overrides.map(override => `${override.condition}: ${override.value}`)]
}

interface TokenRowProps {
  token: DesignToken
  copy: DesignSystemCopy
  revision: string
}

function TokenRow({ copy, revision, token }: TokenRowProps) {
  const ref = useRef<HTMLDivElement>(null)
  const resolved = useResolvedToken(ref, token.name, revision)
  const color = useSwatchColor(ref, resolved, revision)
  const kind = resolved?.kind ?? 'value'

  return (
    <div
      className="ds-token"
      data-kind={kind}
      data-token={token.name}
      ref={ref}
      style={{ '--ds-specimen': `var(${token.name})` } as CSSProperties}
    >
      <span aria-hidden className="ds-token-specimen">
        {SPECIMENS[kind]()}
      </span>
      <div className="ds-token-main">
        <CopyButton appearance="inline" className="ds-token-name" label={copy.copyName(token.name)} text={token.name}>
          {token.name}
        </CopyButton>
        {token.note && <p className="ds-token-note">{token.note}</p>}
        {definedAs(token, copy)
          .filter(line => line !== color && line !== resolved?.value)
          .map((line, index) => (
            <code className="ds-token-defined" key={`${index}:${line}`}>
              {line}
            </code>
          ))}
      </div>
      <code className="ds-token-value">{color || resolved?.value}</code>
    </div>
  )
}

/** Re-resolves every value when what they depend on changes: mode, skin, accent, width class. */
function useTokenRevision(): string {
  const { renderedMode, themeName } = useTheme()
  const accent = useStore($portalAccent)
  const width = useStore($widthClass)

  return `${renderedMode}:${themeName}:${accent}:${width}`
}

export const tokenFamilyId = (file: string) => `ds-tokens-${file}`

export const familyName = (copy: DesignSystemCopy, file: string) => copy.families[file] ?? `${file}.css`

interface TokensSectionProps {
  families: readonly TokenFamily[]
  copy: DesignSystemCopy
}

export function TokensSection({ copy, families }: TokensSectionProps) {
  const revision = useTokenRevision()

  return (
    <>
      {families.map(family => (
        <section
          aria-labelledby={`${tokenFamilyId(family.file)}-title`}
          className="ds-family"
          id={tokenFamilyId(family.file)}
          key={family.file}
        >
          <header className="ds-family-head">
            <h3 id={`${tokenFamilyId(family.file)}-title`}>{familyName(copy, family.file)}</h3>
            <span className="ds-meta">{copy.tokenCount(family.tokens.length)}</span>
            <code className="ds-source">src/app/redesign/tokens/{family.file}.css</code>
            {family.about && <p className="ds-about">{family.about}</p>}
          </header>
          <div className="ds-token-list">
            {family.tokens.map(token => (
              <TokenRow copy={copy} key={token.name} revision={revision} token={token} />
            ))}
          </div>
        </section>
      ))}
    </>
  )
}
