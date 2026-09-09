/**
 * Unit tests for the code-point-aware long-message splitter.
 *
 * WhatsApp counts message length in characters (code points), not UTF-16
 * code units: every symbol, letter, or emoji counts as 1 character.  The
 * splitter must therefore never count an astral character (emoji, Unicode
 * math-alphanumeric glyphs) twice and never split a surrogate pair.
 *
 * Regression: Hermes heading formatting emits astral Unicode glyphs; the
 * legacy UTF-16 splitter counted those at 2x, so a ~3000-char streamed
 * EDIT crossed the 4096 threshold, split, and chunk 2 was surfaced as a
 * brand-new unlinked message — the duplicate-bubble regression.
 */

import { strict as assert } from 'node:assert';

import {
  codePointLength,
  splitLongMessage,
} from './bridge_helpers.js';

// Astral-plane glyph used by Hermes heading formatting (𝐁 = U+1D401,
// 1 code point = 2 UTF-16 code units).
const BOLD_B = '\u{1D401}';

// -- counting -------------------------------------------------------------
{
  assert.equal(codePointLength(''), 0);
  assert.equal(codePointLength('abc'), 3);
  assert.equal(codePointLength('𝐁𝐢𝐠'), 3);
  assert.equal(codePointLength('a𝐁b'), 3);
  assert.equal(codePointLength('😀😀'), 2); // emoji: 1 char each
  console.log('  ✓ codePointLength counts characters, not UTF-16 units');
}

// -- short input never splits --------------------------------------------
{
  assert.deepEqual(splitLongMessage(''), []);
  assert.deepEqual(splitLongMessage('hello'), ['hello']);
  assert.deepEqual(splitLongMessage('x'.repeat(4096)), ['x'.repeat(4096)]);
  console.log('  ✓ short / boundary input stays a single chunk');
}

// -- the regression: astral text within 4096 CHARS must not split ---------
{
  // 3000 astral glyphs = 6000 UTF-16 units: legacy splitter (4096 units)
  // would emit 2 chunks and the /edit handler would surface chunk 2 as a
  // new message.  Code-point counting keeps it one chunk.
  const text = BOLD_B.repeat(3000);
  const chunks = splitLongMessage(text, 4096);
  assert.equal(chunks.length, 1);
  assert.equal(chunks[0], text);
  console.log('  ✓ 3000 astral chars (6000 UTF-16 units) stay one chunk at 4096');
}

// -- oversized astral text splits on code-point boundaries ---------------
{
  const text = BOLD_B.repeat(5000); // 5000 cp > 4096
  const chunks = splitLongMessage(text, 4096);
  assert.ok(chunks.length >= 2, 'oversized text must split');
  for (const chunk of chunks) {
    assert.ok(codePointLength(chunk) <= 4096, 'every chunk fits the char limit');
    assert.ok(chunk.isWellFormed(), 'no lone surrogates in any chunk');
  }
  // Rejoining chunks must reproduce the original exactly.
  assert.equal(chunks.join(''), text);
  console.log('  ✓ oversized astral text splits at 4096 chars, surrogate-safe');
}

// -- whitespace preference preserved -------------------------------------
{
  const text = `${'a'.repeat(100)}\n${'b'.repeat(5000)}`;
  const chunks = splitLongMessage(text, 4096);
  assert.ok(chunks.length >= 2);
  assert.equal(chunks[0].trimEnd(), 'a'.repeat(100)); // broke at the newline
  assert.ok(chunks.every((c) => c.isWellFormed()));
  console.log('  ✓ newline boundary still preferred');
}

{
  // Space fallback when no newline in the window.
  const text = `${'a'.repeat(4000)} ${'b'.repeat(4000)}`;
  const chunks = splitLongMessage(text, 4096);
  assert.ok(chunks.length >= 2);
  assert.ok(chunks.every((c) => c.isWellFormed()));
  console.log('  ✓ space boundary fallback still works');
}

{
  // Hard break when no whitespace at all.
  const text = 'x'.repeat(9000);
  const chunks = splitLongMessage(text, 4096);
  assert.ok(chunks.length >= 3);
  assert.ok(chunks.every((c) => codePointLength(c) <= 4096));
  assert.equal(chunks.join(''), text);
  console.log('  ✓ hard break on unbroken text is lossless');
}

console.log('\nAll splitLongMessage tests passed.');
