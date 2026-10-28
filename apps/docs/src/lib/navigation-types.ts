export interface NavigationLink {
  type: 'link';
  label: string;
  href: string;
}

export interface NavigationDoc {
  type: 'doc';
  id: string;
  label?: string;
}

export interface NavigationCategory {
  type: 'category';
  label: string;
  key?: string;
  collapsed?: boolean;
  collapsible?: boolean;
  link?: { type: 'doc'; id: string };
  items: NavigationItem[];
}

export type NavigationItem = string | NavigationLink | NavigationDoc | NavigationCategory;

export interface NavigationConfig {
  docs: NavigationItem[];
}
