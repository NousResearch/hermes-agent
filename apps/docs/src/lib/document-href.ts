import { localizedHref, splitLocalePath, type Locale } from './urls.ts';

export type FragmentTranslations = Record<string, Record<string, string | null>>;

/** Translate only verified section pairs; null retains the exact English target.
 * Unknown fragments remain untouched so the export validator still rejects them. */
export function documentHref(href: string, locale: Locale, pathname: string, translations: FragmentTranslations): string {
  const localized = localizedHref(href, locale);
  if (locale !== 'zh-Hans' || /^(?:[a-z][a-z\d+.-]*:|\/\/)/i.test(href)) return localized;
  const hashAt = href.indexOf('#');
  if (hashAt === -1) return localized;
  const before = href.slice(0, hashAt);
  const fragment = href.slice(hashAt + 1);
  let id: string;
  try {
    id = decodeURIComponent(fragment);
  } catch {
    return localized;
  }
  const logical = splitLocalePath(before || pathname).path;
  const route = logical.split('?', 1)[0].replace(/\/+$/, '') || '/';
  const mapping = translations[route];
  if (!mapping || !Object.hasOwn(mapping, id)) return localized;
  const translated = mapping[id];
  if (translated === null) return `${localizedHref(before || pathname, 'en')}#${fragment}`;
  return `${before ? localizedHref(before, locale) : ''}#${encodeURIComponent(translated)}`;
}
