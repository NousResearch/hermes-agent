import './design-system.css'

import { type RefObject, useEffect, useRef, useState } from 'react'
import { Navigate } from 'react-router'

import { tokenFamilies } from '@/lib/design-tokens'
import { usePortalShell } from '@/store/shell-variant'

import { PageSidebarCaption, PageSidebarList, PageSidebarPanel, PageSidebarRow } from '../../overlays/page-sidebar'
import { NEW_CHAT_ROUTE } from '../../routes'
import { closeOpenSheet } from '../../shell/shell-sheet'
import { PageHeader } from '../page-header'

import { componentGroupId, ComponentsSection, NOT_SHOWN_ID } from './components-section'
import { type DesignSystemCopy, useDesignSystemCopy } from './design-system-copy'
import { PreviewControls } from './preview-controls'
import { GALLERY_GROUPS } from './registry'
import { STILL_TO_UNIFY, TOKEN_SOURCES } from './sources'
import { familyName, tokenFamilyId, TokensSection } from './tokens-section'
import { UnifySection } from './unify-section'

const FAMILIES = tokenFamilies(TOKEN_SOURCES)

const UNIFY_ID = 'ds-still-to-unify'

interface NavGroup {
  /** Absent for a group whose one row already names it. */
  caption?: string
  rows: { id: string; label: string }[]
}

function navGroups(copy: DesignSystemCopy): NavGroup[] {
  return [
    {
      caption: copy.tokens,
      rows: FAMILIES.map(family => ({ id: tokenFamilyId(family.file), label: familyName(copy, family.file) }))
    },
    {
      caption: copy.components,
      rows: [
        ...GALLERY_GROUPS.map(group => ({ id: componentGroupId(group), label: copy.groups[group] })),
        { id: NOT_SHOWN_ID, label: copy.notShown }
      ]
    },
    { rows: [{ id: UNIFY_ID, label: copy.stillToUnify }] }
  ]
}

/** The section at the top of the stage: the last one in the stage's top third. */
function useCurrentSection(stage: RefObject<HTMLDivElement | null>, ids: readonly string[]): string {
  const [current, setCurrent] = useState(ids[0] ?? '')
  const key = ids.join(' ')

  useEffect(() => {
    const root = stage.current

    if (!root) {
      return
    }

    const visible = new Set<string>()

    const observer = new IntersectionObserver(
      entries => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            visible.add(entry.target.id)
          } else {
            visible.delete(entry.target.id)
          }
        }

        // The band holds the end of one section and the start of the next; the later one is current.
        const last = key
          .split(' ')
          .filter(id => visible.has(id))
          .at(-1)

        if (last) {
          setCurrent(last)
        }
      },
      { root, rootMargin: '0px 0px -66% 0px' }
    )

    for (const id of key.split(' ')) {
      const section = document.getElementById(id)

      if (section) {
        observer.observe(section)
      }
    }

    return () => observer.disconnect()
  }, [key, stage])

  return current
}

function DesignSystemPage() {
  const copy = useDesignSystemCopy()
  const stage = useRef<HTMLDivElement>(null)
  const groups = navGroups(copy)

  const current = useCurrentSection(
    stage,
    groups.flatMap(group => group.rows.map(row => row.id))
  )

  // Navigation never animates (DESIGN.md › Portal design language): jump, and close the sheet below wide.
  const go = (id: string) => {
    document.getElementById(id)?.scrollIntoView({ block: 'start' })
    closeOpenSheet()
  }

  return (
    <section aria-label={copy.title} className="ds-page" data-page-surface="">
      <PageSidebarPanel title={copy.contents}>
        <PageSidebarList>
          {groups.map(group => (
            <div className="ds-nav-group" key={group.rows[0].id}>
              {group.caption && <PageSidebarCaption>{group.caption}</PageSidebarCaption>}
              {group.rows.map(row => (
                <PageSidebarRow current={row.id === current} key={row.id} onClick={() => go(row.id)}>
                  {row.label}
                </PageSidebarRow>
              ))}
            </div>
          ))}
        </PageSidebarList>
      </PageSidebarPanel>

      <div className="ds-stage page-stage" data-scrolls="" ref={stage}>
        <PageHeader description={copy.description} title={copy.title} />
        <PreviewControls copy={copy} />

        <h2 className="ds-part-title">{copy.tokens}</h2>
        <p className="ds-about">{copy.tokensIntro}</p>
        <TokensSection copy={copy} families={FAMILIES} />

        <h2 className="ds-part-title">{copy.components}</h2>
        <p className="ds-about">{copy.componentsIntro}</p>
        <ComponentsSection copy={copy} />

        <section aria-labelledby={`${UNIFY_ID}-title`} className="ds-family" id={UNIFY_ID}>
          <h2 className="ds-part-title" id={`${UNIFY_ID}-title`}>
            {copy.stillToUnify}
          </h2>
          <p className="ds-about">{copy.stillToUnifyIntro}</p>
          <UnifySection blocks={STILL_TO_UNIFY} />
        </section>
      </div>
    </section>
  )
}

/** `#/design-system`: the redesign's tokens and primitives, rendered from their source. It documents
 *  the redesign's language, whose tokens exist only under it, so classic goes to a new chat. */
export function DesignSystemView() {
  return usePortalShell() ? <DesignSystemPage /> : <Navigate replace to={NEW_CHAT_ROUTE} />
}
