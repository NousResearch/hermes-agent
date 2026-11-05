import type { Metadata } from 'next';
import { canonicalUrl, localizedHref, type Locale } from './urls.ts';

export function catalogAlternates(path: string, locale: Locale): Metadata['alternates'] {
  return {
    canonical: canonicalUrl(localizedHref(path, locale)),
    languages: {
      en: canonicalUrl(localizedHref(path, 'en')),
      'zh-Hans': canonicalUrl(localizedHref(path, 'zh-Hans')),
      'x-default': canonicalUrl(localizedHref(path, 'en')),
    },
  };
}
