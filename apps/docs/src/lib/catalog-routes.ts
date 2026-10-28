import type { Metadata } from 'next';
import { buildPluginPageData, decodeCatalogParam, getAuthorPageData, loadCatalogSnapshot } from './plugin-pages.ts';
import { authorPagePath, authorSeoDescription, pluginPagePath, pluginSeoDescription } from '../components/PluginCatalog/catalog.ts';
import { canonicalUrl, localizedHref, type Locale } from './urls.ts';

export function catalogAlternates(path: string, locale: Locale): Metadata['alternates'] {
  return {
    canonical: canonicalUrl(localizedHref(path, locale)),
    languages: {
      en: canonicalUrl(localizedHref(path, 'en')),
      'zh-Hans': canonicalUrl(localizedHref(path, 'zh-Hans')),
    },
  };
}

/** Metadata needs the catalog record, never the remote README body. */
export async function pluginPageMetadata(
  { params }: { params: Promise<{ name: string }> },
  locale: Locale,
): Promise<Metadata> {
  const { name } = await params;
  const data = buildPluginPageData(await loadCatalogSnapshot(), decodeCatalogParam(name));
  if (!data) return {};
  const alternates = catalogAlternates(pluginPagePath(data.plugin.name), locale);
  return {
    title: `${data.plugin.name} · Plugin Catalog`,
    description: pluginSeoDescription(data.plugin),
    alternates,
    openGraph: {
      type: 'website',
      url: canonicalUrl(localizedHref(pluginPagePath(data.plugin.name), locale)),
      images: data.plugin.image ? [data.plugin.image] : undefined,
    },
  };
}

export async function authorPageMetadata(
  { params }: { params: Promise<{ author: string }> },
  locale: Locale,
): Promise<Metadata> {
  const { author } = await params;
  const data = await getAuthorPageData(decodeCatalogParam(author));
  if (!data) return {};
  return {
    title: `Plugins by ${data.author.name} · Plugin Catalog`,
    description: authorSeoDescription(data.author, data.plugins.length),
    alternates: catalogAlternates(authorPagePath(data.author.slug), locale),
  };
}
