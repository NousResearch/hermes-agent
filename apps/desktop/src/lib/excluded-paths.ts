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

// #55169: the file tree's "Show gitignored files" toggle is an explicit user
// request for the whole project, but it could never reveal entries this set
// drops unconditionally — deny-by-default .gitignore layouts rely on names
// like `out/` or `vendor/` for real build output, and users could not browse
// them even with the eye toggle on. Names here stay hidden even when the
// project opted in; they are the noise the transports strip as well
// (FS_READDIR_HIDDEN in electron/fs-read-dir.ts, _FS_READDIR_HIDDEN in
// hermes_cli/web_routers/files.py), so the toggle's reveal floor must not go
// below what either transport can return. Keep the two transport sets in sync
// with this floor: a name only ever filtered by ALWAYS_EXCLUDED (out, vendor,
// coverage, .DS_Store, …) belongs here so a synced project can reveal it; the
// names both transports strip (.git internals, dep/build/cache dirs) do not.
export const SHOW_IGNORED_EXCLUDED = new Set([
  '.git',
  '.hg',
  '.svn',
  'node_modules',
  'bower_components',
  '.venv',
  'venv',
  '__pycache__',
  '.mypy_cache',
  '.pytest_cache',
  '.ruff_cache',
  '.tox',
  '.gradle',
  '.idea',
  'dist',
  'build',
  'target',
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
  '.DS_Store',
  'Thumbs.db'
])

// True when any segment of a relative path is excluded (review rows like
// `node_modules/.bin/foo` or a bare `.DS_Store`). Handles `/` and `\`.
export const isExcludedPath = (relPath: string): boolean => relPath.split(/[/\\]/).some(seg => ALWAYS_EXCLUDED.has(seg))
