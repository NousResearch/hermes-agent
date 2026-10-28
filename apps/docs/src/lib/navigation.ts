import navigation from './navigation-data';
import { source } from './source';
import { buildNavigationTree } from './navigation-tree';
import type { NavigationItem } from './navigation-types';
import type { Locale } from './urls';

export function getNavigationTree(locale: Locale) {
  return buildNavigationTree(navigation.docs as NavigationItem[], source.getPages(locale), locale);
}
