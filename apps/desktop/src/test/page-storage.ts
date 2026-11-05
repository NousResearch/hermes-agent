/**
 * The object that holds `window.localStorage`'s methods, for spying on the page's storage reads
 * and writes. On Node 22 that is jsdom's `Storage.prototype`: a spy defined on the instance is kept
 * as a stored item and never intercepts a call. From Node 25 the test setup installs a plain
 * in-memory storage (see vitest.setup.ts) whose methods are its own and whose prototype has none.
 */
export function pageStorage(): Storage {
  const storage = window.localStorage

  return Object.prototype.hasOwnProperty.call(storage, 'setItem')
    ? storage
    : (Object.getPrototypeOf(storage) as Storage)
}
