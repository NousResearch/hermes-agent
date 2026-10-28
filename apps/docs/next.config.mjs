import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { createMDX } from 'fumadocs-mdx/next';

const appDir = path.dirname(fileURLToPath(import.meta.url));
const withMDX = createMDX();

export default withMDX({
  output: 'export',
  agentRules: false,
  basePath: '/docs',
  trailingSlash: true,
  images: { unoptimized: true },
  reactStrictMode: true,
  serverExternalPackages: ['shiki', 'fumadocs-mdx'],
  outputFileTracingRoot: path.resolve(appDir, '../..'),
  turbopack: { root: path.resolve(appDir, '../..') },
});
