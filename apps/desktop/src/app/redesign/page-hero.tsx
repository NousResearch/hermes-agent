import './page-hero.css'

/** The baked hero artworks (`page-hero.css` maps each to its assets and geometry): `tide` the agent
 *  page (18563:12855), `stream` Account & Plan (18563:12252), `veil` Machines (18563:12520). */
export type PageHeroArt = 'stream' | 'tide' | 'veil'

/** The band of artwork behind the top of a page's stage (DESIGN.md › Page hero): purely
 *  decorative, so it is hidden from assistive tech and takes no pointer. Render it as the first
 *  child of the stage after any page sidebar panel; it fills the stage's width from
 *  `--page-stage-nav` and fades out by itself, and the content after it paints over it. */
export function PageHero({ art }: { art: PageHeroArt }) {
  return <div aria-hidden className="portal-page-hero" data-art={art} />
}
