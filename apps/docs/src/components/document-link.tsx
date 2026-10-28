"use client";

import Link from 'next/link';
import { useI18n } from 'fumadocs-ui/contexts/i18n';
import type { ComponentProps } from 'react';
import { localizedHref } from '@/lib/urls';

export function DocumentLink({ href = '', children, ...props }: ComponentProps<'a'>) {
  const { locale } = useI18n();
  if (/^(?:[a-z][a-z\d+.-]*:|\/\/|#)/i.test(href)) {
    const external = /^(?:https?:|\/\/)/i.test(href);
    return <a {...props} href={href} target={external ? (props.target ?? '_blank') : props.target} rel={external ? 'noreferrer noopener' : props.rel}>{children}</a>;
  }
  return <Link {...props} href={localizedHref(href, locale === 'zh-Hans' ? locale : 'en')}>{children}</Link>;
}
