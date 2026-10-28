import type { Root, Node, Item } from 'fumadocs-core/page-tree';
import type { NavigationItem } from './navigation-types';
import { localizedHref, type Locale } from './urls.ts';

interface NavigationPage {
  path: string;
  url: string;
  data: { title: string; sidebar_label?: string };
}

export function buildNavigationTree(items: NavigationItem[], pages: NavigationPage[], locale: Locale): Root {
  const byId = new Map(pages.map((page) => [page.path.replace(/^(en|zh-Hans)\//, '').replace(/\.mdx?$/, ''), page]));
  function document(id: string, label?: string): Item {
    const page = byId.get(id);
    if (!page) throw new Error(`Sidebar document does not exist: ${id}`);
    return { type: 'page', name: label ?? page.data.sidebar_label ?? page.data.title, url: page.url };
  }
  function node(item: NavigationItem): Node {
    if (typeof item === 'string') return document(item);
    if (item.type === 'doc') return document(item.id, item.label);
    if (item.type === 'link') return { type: 'page', name: item.label, url: localizedHref(item.href, locale) };
    return {
      type: 'folder',
      name: item.label,
      $id: item.key ?? item.label,
      defaultOpen: item.collapsed === false,
      index: item.link ? document(item.link.id) : undefined,
      children: item.items.map(node),
    };
  }
  return { name: 'Hermes Agent', children: items.map(node) };
}
