#!/usr/bin/env node
// validate_shapes.mjs — verify that the note/text/frame records this skill
// documents are valid against the real tldraw SDK v5 schema.
//
// Usage:
//   npm install @tldraw/tlschema
//   node validate_shapes.mjs
//
// Exits 0 when all sample records validate, 1 otherwise. No network, no DOM.

import { createTLSchema, toRichText, createShapeId, PageRecordType } from '@tldraw/tlschema'

const schema = createTLSchema()
const shapeRecord = schema.types.shape
const pageId = PageRecordType.createId()

// Complete default prop sets (required when building raw records outside the editor).
const COMPLETE = {
  note: {
    richText: toRichText(''), color: 'black', labelColor: 'black', size: 'm',
    font: 'draw', align: 'middle', verticalAlign: 'middle', growY: 0,
    fontSizeAdjustment: 0, url: '', scale: 1, textLastEditedBy: '',
  },
  text: {
    richText: toRichText(''), color: 'black', size: 'm', font: 'draw',
    textAlign: 'start', w: 8, scale: 1, autoSize: true,
  },
  frame: { w: 300, h: 640, name: '', color: 'black' },
}

function makeRecord(type, props, meta = {}, x = 0, y = 0) {
  return {
    id: createShapeId(), typeName: 'shape', type, parentId: pageId, index: 'a1',
    x, y, rotation: 0, isLocked: false, opacity: 1, meta,
    props: { ...COMPLETE[type], ...props },
  }
}

const cases = [
  ['frame', makeRecord('frame', { w: 300, h: 640, name: 'To Do' }, { role: 'column' })],
  ['text',  makeRecord('text',  { richText: toRichText('To Do  ·  WIP 2'), size: 's', color: 'grey', font: 'sans' }, { role: 'count' }, 8, -34)],
  ['note',  makeRecord('note',  { richText: toRichText('Design the thing'), size: 's' }, { role: 'card' }, 20, 48)],
]

let ok = 0
const validated = []
for (const [name, rec] of cases) {
  try {
    validated.push(shapeRecord.validate(rec))
    console.log(`OK   ${name}`)
    ok++
  } catch (e) {
    console.log(`FAIL ${name}: ${String(e.message).split('\n')[0]}`)
  }
}

// Round-trip through JSON to mimic file save/load.
let rok = 0
for (const v of validated) {
  try { shapeRecord.validate(JSON.parse(JSON.stringify(v))); rok++ } catch { /* counted below */ }
}

console.log(`\n${ok}/${cases.length} shape records valid against the tldraw schema.`)
console.log(`${rok}/${validated.length} survive a JSON round-trip (file save/load).`)
process.exit(ok === cases.length && rok === validated.length ? 0 : 1)
