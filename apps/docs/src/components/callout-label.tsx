"use client";

import { useI18n } from 'fumadocs-ui/contexts/i18n';
import { calloutLabel } from '@/remark/admonition-labels.mjs';

/** Untitled callouts read their default label in the locale being viewed. */
export function DefaultCalloutLabel({ type }: { type: string }) {
  const { locale } = useI18n();
  return calloutLabel(type, locale === 'zh-Hans' ? 'zh-Hans' : 'en');
}
