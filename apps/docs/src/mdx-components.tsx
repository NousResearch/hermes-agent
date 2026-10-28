import defaultComponents from 'fumadocs-ui/mdx';
import type { MDXComponents } from 'mdx/types';
import { DocumentLink } from '@/components/document-link';
import { Mermaid } from '@/components/mermaid';
import UserStoriesCollage from '@/components/UserStoriesCollage';
import AutomationBlueprintsCatalog from '@/components/AutomationBlueprintsCatalog';
import { publicHref } from '@/lib/urls';

export function getMDXComponents(components?: MDXComponents): MDXComponents {
  return {
    ...defaultComponents,
    a: DocumentLink,
    Link: DocumentLink,
    UserStoriesCollage,
    AutomationBlueprintsCatalog,
    img: ({ src, alt, ...props }) => <img {...props} alt={alt ?? ''} src={typeof src === 'string' && src.startsWith('/') ? publicHref(src) : src} />,
    Mermaid,
    ...components,
  };
}

export const useMDXComponents = getMDXComponents;
