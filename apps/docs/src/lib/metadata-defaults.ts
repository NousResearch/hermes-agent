import type { Metadata } from 'next';
import { publicHref, SITE_URL, type Locale } from './urls.ts';

// Next replaces OpenGraph per route segment rather than merging its fields.
const SITE_OPEN_GRAPH = {
  siteName: 'Hermes Agent',
  images: [publicHref('/img/hermes-agent-banner.png')],
};

// Omit twitter:image so a page-specific og:image stays authoritative.
const SITE_TWITTER = { card: 'summary_large_image' as const };

type OpenGraphFields = Partial<Extract<NonNullable<Metadata['openGraph']>, { type?: 'website' }>>;

/** Preserve the site identity unless a page supplies its own banner. */
export function pageOpenGraph(fields: OpenGraphFields = {}) {
  const { images, ...pageFields } = fields;
  return { ...SITE_OPEN_GRAPH, ...pageFields, images: images ?? SITE_OPEN_GRAPH.images };
}

/** Root-layout metadata shared by both locale trees. */
export function siteMetadata(locale: Locale): Metadata {
  return {
    metadataBase: new URL(SITE_URL),
    title: { default: 'Hermes Agent Documentation', template: '%s | Hermes Agent' },
    description: 'The self-improving AI agent built by Nous Research.',
    icons: { icon: publicHref('/img/favicon.ico'), apple: publicHref('/img/apple-touch-icon.png') },
    openGraph: SITE_OPEN_GRAPH,
    twitter: SITE_TWITTER,
    // Preserve the existing DocSearch crawler facets during the renderer cutover.
    other: {
      'docsearch:language': locale,
      'docsearch:version': 'current',
      'docsearch:docusaurus_tag': 'docs-default-current',
      docusaurus_locale: locale,
      docusaurus_version: 'current',
      docusaurus_tag: 'docs-default-current',
    },
  };
}
