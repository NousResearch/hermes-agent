import type { Metadata } from 'next';
import { canonicalUrl, LOCALES, type Locale } from './urls.ts';

interface MetadataPage {
  url: string;
  data: { title: string; description?: string; seo_title?: string };
}

export function documentPageMetadata(
  page: MetadataPage,
  translations: Partial<Record<Locale, MetadataPage>>,
): Metadata {
  const title = page.data.seo_title ?? page.data.title;
  return {
    title: page.data.seo_title ? { absolute: title } : title,
    description: page.data.description,
    alternates: {
      canonical: canonicalUrl(page.url),
      languages: {
        ...Object.fromEntries(LOCALES.map((locale) => [
          locale, canonicalUrl(translations[locale]?.url ?? page.url),
        ])),
        'x-default': canonicalUrl(translations.en?.url ?? page.url),
      },
    },
    openGraph: { title, description: page.data.description, url: canonicalUrl(page.url) },
  };
}
