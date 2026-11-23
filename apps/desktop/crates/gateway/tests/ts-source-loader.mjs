// Resolve the original source's ESM .js imports to neighboring .ts files.
// Node's built-in type stripper executes them; source content is not rewritten.
export async function resolve(specifier, context, nextResolve) {
  try { return await nextResolve(specifier, context) }
  catch (error) {
    if (specifier.endsWith('.js') && context.parentURL?.endsWith('.ts')) {
      return nextResolve(specifier.slice(0, -3) + '.ts', context)
    }
    throw error
  }
}
