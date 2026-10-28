import { defineI18n } from 'fumadocs-core/i18n';

export const i18n = defineI18n({
  defaultLanguage: 'en',
  languages: ['en', 'zh-Hans'],
  parser: 'dir',
  hideLocale: 'default-locale',
  fallbackLanguage: 'en',
});
