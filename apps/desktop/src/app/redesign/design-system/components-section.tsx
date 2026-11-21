import type { DesignSystemCopy } from './design-system-copy'
import { GALLERY, GALLERY_GROUPS, type GalleryEntry, type GalleryGroup, NOT_SHOWN } from './registry'
import type { GalleryExample } from './registry/types'

export const componentGroupId = (group: GalleryGroup) => `ds-components-${group}`

export const NOT_SHOWN_ID = 'ds-components-not-shown'

/** Mounts an example as a component of its own, so its hooks are its own. */
function Example({ example }: { example: GalleryExample }) {
  const Specimen = example.render

  return (
    <figure className="ds-example">
      <div className="ds-example-stage">
        <Specimen />
      </div>
      <figcaption>{example.label}</figcaption>
    </figure>
  )
}

function EntryCard({ entry }: { entry: GalleryEntry }) {
  return (
    <article aria-label={entry.title} className="ds-entry" data-entry={entry.source}>
      <header className="ds-entry-head">
        <h4>{entry.title}</h4>
        <code className="ds-source">src/{entry.source}</code>
        <p className="ds-about">{entry.use}</p>
      </header>
      <div className="ds-examples" data-dense={entry.dense ? '' : undefined}>
        {entry.examples.map(example => (
          <Example example={example} key={example.label} />
        ))}
      </div>
    </article>
  )
}

export function ComponentsSection({ copy }: { copy: DesignSystemCopy }) {
  return (
    <>
      {GALLERY_GROUPS.map(group => (
        <section
          aria-labelledby={`${componentGroupId(group)}-title`}
          className="ds-family"
          id={componentGroupId(group)}
          key={group}
        >
          <header className="ds-family-head">
            <h3 id={`${componentGroupId(group)}-title`}>{copy.groups[group]}</h3>
          </header>
          {GALLERY.filter(entry => entry.group === group).map(entry => (
            <EntryCard entry={entry} key={entry.title} />
          ))}
        </section>
      ))}
      <section aria-labelledby={`${NOT_SHOWN_ID}-title`} className="ds-family" id={NOT_SHOWN_ID}>
        <header className="ds-family-head">
          <h3 id={`${NOT_SHOWN_ID}-title`}>{copy.notShown}</h3>
          <p className="ds-about">{copy.notShownIntro}</p>
        </header>
        <dl className="ds-not-shown">
          {Object.entries(NOT_SHOWN).map(([file, reason]) => (
            <div key={file}>
              <dt>
                <code>{file}</code>
              </dt>
              <dd>{reason}</dd>
            </div>
          ))}
        </dl>
      </section>
    </>
  )
}
