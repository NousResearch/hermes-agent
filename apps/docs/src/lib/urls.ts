export const SITE_URL = 'https://hermes-agent.nousresearch.com';
export const DOCS_BASE_PATH = '/docs';
export const LOCALES = ['en', 'zh-Hans'] as const;
export type Locale = (typeof LOCALES)[number];

function isOutsidePath(href: string): boolean {
  return /^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href);
}

function hasPrefix(path: string, prefix: string): boolean {
  return path === prefix || path.startsWith(`${prefix}/`);
}

/** Return the logical page path, independent of its deployment and language. */
export function splitLocalePath(href: string): { locale: Locale; path: string } {
  const suffixIndex = href.search(/[?#]/);
  const suffix = suffixIndex < 0 ? '' : href.slice(suffixIndex);
  let path = suffixIndex < 0 ? href : href.slice(0, suffixIndex);
  if (hasPrefix(path, DOCS_BASE_PATH)) path = path.slice(DOCS_BASE_PATH.length);
  const locale = hasPrefix(path, '/zh-Hans') ? 'zh-Hans' : 'en';
  if (locale === 'zh-Hans') path = path.slice('/zh-Hans'.length);
  return { locale, path: (path || '/') + suffix };
}

/** Next links own basePath; only the language belongs in their href. */
export function localizedHref(href: string, locale: Locale = 'en'): string {
  if (isOutsidePath(href)) return href;
  const { path } = splitLocalePath(href.startsWith('/') ? href : `/${href}`);
  return locale === 'en' ? path : `/zh-Hans${path === '/' ? '' : path}`;
}

/** Native anchors, images and fetch do not receive Next's automatic basePath. */
export function publicHref(href: string): string {
  if (isOutsidePath(href)) return href;
  const path = href.startsWith('/') ? href : `/${href}`;
  return hasPrefix(path.split(/[?#]/, 1)[0], DOCS_BASE_PATH) ? path : `${DOCS_BASE_PATH}${path}`;
}

export function canonicalUrl(href: string): string {
  return new URL(publicHref(href), SITE_URL).href;
}
