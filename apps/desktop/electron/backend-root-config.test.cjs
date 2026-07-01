const test = require('node:test')
const assert = require('node:assert/strict')
const os = require('node:os')
const path = require('node:path')

const { parseDesktopBackendRoot } = require('./backend-root-config.cjs')

const HOME = os.homedir()

// --- valid: nested backend_root under a top-level desktop: block ---
test('valid nested scalar returns the path', () => {
  const t = 'model: gpt\ndesktop:\n  backend_root: /Users/x/.hermes/runtime/hermes-agent\n'
  assert.equal(parseDesktopBackendRoot(t), '/Users/x/.hermes/runtime/hermes-agent')
})

// --- different top-level block with a backend_root key must be IGNORED ---
test('backend_root under a DIFFERENT block is ignored', () => {
  const t = 'other:\n  backend_root: /wrong/tree\ndesktop:\n  something: 1\n'
  assert.equal(parseDesktopBackendRoot(t), null)
})

// --- commented-out key never matches ---
test('commented-out backend_root is ignored', () => {
  const t = 'desktop:\n  # backend_root: /old/tree\n'
  assert.equal(parseDesktopBackendRoot(t), null)
})

// --- trailing inline comment is stripped ---
test('trailing comment is stripped', () => {
  const t = 'desktop:\n  backend_root: /a/b   # a note\n'
  assert.equal(parseDesktopBackendRoot(t), '/a/b')
})

// --- quoted scalars (single + double) unwrap ---
test('double-quoted scalar unwraps', () => {
  assert.equal(parseDesktopBackendRoot('desktop:\n  backend_root: "/a b/c"\n'), '/a b/c')
})
test('single-quoted scalar unwraps', () => {
  assert.equal(parseDesktopBackendRoot("desktop:\n  backend_root: '/a b/c'\n"), '/a b/c')
})

// --- ~ and $HOME expansion ---
test('~ expands to home', () => {
  assert.equal(parseDesktopBackendRoot('desktop:\n  backend_root: ~/rt\n'), path.join(HOME, 'rt'))
})
test('$HOME expands to home', () => {
  assert.equal(parseDesktopBackendRoot('desktop:\n  backend_root: $HOME/rt\n'), path.join(HOME, 'rt'))
})

// --- empty value -> null (auto-resolve) ---
test('empty value returns null', () => {
  assert.equal(parseDesktopBackendRoot('desktop:\n  backend_root:\n'), null)
  assert.equal(parseDesktopBackendRoot('desktop:\n  backend_root: ""\n'), null)
})

// --- multi-document: scan stops at first --- and only reads doc 1 ---
test('multi-document: key after --- is not read', () => {
  const t = 'desktop:\n  other: 1\n---\ndesktop:\n  backend_root: /doc2\n'
  assert.equal(parseDesktopBackendRoot(t), null)
})
test('multi-document: key in first doc IS read', () => {
  const t = 'desktop:\n  backend_root: /doc1\n---\nother: 2\n'
  assert.equal(parseDesktopBackendRoot(t), '/doc1')
})
// --- leading `---` document-start marker (valid YAML, emitted by Helm/Ansible/etc.) is NOT a separator ---
test('leading --- document-start marker: override under it IS read', () => {
  const t = '---\ndesktop:\n  backend_root: /leadmarker\n'
  assert.equal(parseDesktopBackendRoot(t), '/leadmarker')
})
test('leading --- after only comments/blanks: override still read', () => {
  const t = '# config\n\n---\ndesktop:\n  backend_root: /aftercomment\n'
  assert.equal(parseDesktopBackendRoot(t), '/aftercomment')
})

// --- tab-indented key -> reject (fail-safe to auto-resolve) ---
test('tab-indented backend_root is rejected', () => {
  const t = 'desktop:\n\tbackend_root: /tabbed\n'
  assert.equal(parseDesktopBackendRoot(t), null)
})

// --- flow-style -> reject ---
test('flow-style desktop block is rejected', () => {
  assert.equal(parseDesktopBackendRoot('desktop: {backend_root: /x}\n'), null)
})

// --- duplicate keys -> first wins ---
test('duplicate keys: first wins', () => {
  const t = 'desktop:\n  backend_root: /first\n  backend_root: /second\n'
  assert.equal(parseDesktopBackendRoot(t), '/first')
})

// --- backend_root at column 0 (not nested) is ignored ---
test('top-level backend_root (not under desktop) is ignored', () => {
  assert.equal(parseDesktopBackendRoot('backend_root: /x\n'), null)
})

// --- no desktop block at all -> null ---
test('absent desktop block returns null', () => {
  assert.equal(parseDesktopBackendRoot('model: gpt\n'), null)
})

// --- empty / non-string input -> null, never throws ---
test('empty or non-string input returns null safely', () => {
  assert.equal(parseDesktopBackendRoot(''), null)
  assert.equal(parseDesktopBackendRoot(null), null)
  assert.equal(parseDesktopBackendRoot(undefined), null)
})

// --- a sibling key after desktop: closes the block (backend_root under next block ignored) ---
test('backend_root after the desktop block closes is ignored', () => {
  const t = 'desktop:\n  foo: 1\nother:\n  backend_root: /wrong\n'
  assert.equal(parseDesktopBackendRoot(t), null)
})
