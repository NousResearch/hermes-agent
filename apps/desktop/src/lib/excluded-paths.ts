// Always hidden across the file tree and review (git) tree, regardless of
// .gitignore: the VCS internals, heavyweight dep/build/cache dirs, and OS noise.
// These bloat both trees and are never worth browsing or reviewing — even in
// repos that track them, and in plain non-git folders.
export const ALWAYS_EXCLUDED = new Set([
  '.git',
  '.hg',
  '.svn',
  'node_modules',
  'bower_components',
  '.venv',
  'venv',
  'env',
  '__pycache__',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
  '.tox',
  '.gradle',
  '.idea',
  'dist',
  'build',
  'out',
  'target',
  'vendor',
  'Pods',
  '.next',
  '.nuxt',
  '.svelte-kit',
  '.output',
  '.turbo',
  '.parcel-cache',
  '.cache',
  '.terraform',
  '.expo',
  '.angular',
  'coverage',
  '.DS_Store',
  'Thumbs.db'
])

// True when any segment of a relative path is excluded (review rows like
// `node_modules/.bin/foo` or a bare `.DS_Store`). Handles `/` and `\`.
export const isExcludedPath = (relPath: string): boolean =>
  relPath.split(/[/\\]/).some(seg => ALWAYS_EXCLUDED.has(seg))
