import type { ReactNode } from 'react';
import { SiteRoot, siteMetadata } from '@/components/site-root';

export const metadata = siteMetadata('zh-Hans');
export default function Layout({ children }: { children: ReactNode }) {
  return <SiteRoot locale="zh-Hans">{children}</SiteRoot>;
}
