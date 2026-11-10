"use client";

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useI18n } from 'fumadocs-ui/contexts/i18n';
import type { ComponentProps } from 'react';
import { documentHref } from '@/lib/document-href';
import fragmentTranslations from '@/lib/fragment-translations.json';

export function DocumentLink({ href = '', children, ...props }: ComponentProps<'a'>) {
  const { locale } = useI18n();
  const pathname = usePathname();
  const destination = documentHref(href, locale === 'zh-Hans' ? locale : 'en', pathname ?? '/', fragmentTranslations);
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(destination)) {
    const external = /^(?:https?:|\/\/)/i.test(destination);
    return <a {...props} href={destination} target={external ? (props.target ?? '_blank') : props.target} rel={external ? 'noreferrer noopener' : props.rel}>{children}</a>;
  }
  return <Link {...props} href={destination}>{children}</Link>;
}
