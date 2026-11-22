// Node ESM resolve hook: let extensionless relative imports resolve to their
// sibling `.ts` module. The desktop electron tests import `'./foo'` while the
// sources are `'./foo.ts'`; plain `node --test` cannot resolve that, so the
// whole electron suite was unrunnable. Node's type stripping reads the `.ts`
// file once the specifier carries the extension.
import { existsSync } from 'node:fs';

const TS_RE = /\.ts$/;

export async function resolve(specifier, context, nextResolve) {
  const isRelative = specifier.startsWith('./') || specifier.startsWith('../');
  if (isRelative && context.parentURL && !TS_RE.test(specifier)) {
    const candidate = new URL(specifier + '.ts', context.parentURL);
    if (existsSync(candidate)) {
      return { url: candidate.href, shortCircuit: true };
    }
  }
  return nextResolve(specifier, context);
}
