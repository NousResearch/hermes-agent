/**
 * Type stub for node-llama-cpp — the migration script does not use local
 * embedding but TypeScript still resolves the module through transitive
 * imports (sqlite.ts → embedding.ts → import("node-llama-cpp")).
 *
 * This stub satisfies the compiler without requiring the actual package.
 */
declare module "node-llama-cpp" {
  const _: any;
  export = _;
}
