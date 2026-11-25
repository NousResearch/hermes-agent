import { docs } from '../../.source/dynamic';
import { loader } from 'fumadocs-core/source';
import { documentSlugs } from './content.mjs';
import { i18n } from './i18n';

export const source = loader({
  baseUrl: '/',
  source: docs.toFumadocsSource(),
  i18n,
  slugs: (file, next) => typeof file.data.slug === 'string'
    ? documentSlugs(file.path.replace(/^(en|zh-Hans)\//, ''), file.data.slug)
    : next(),
});

export type DocPage = ReturnType<typeof source.getPages>[number];
