import test from 'node:test';
import assert from 'node:assert/strict';

import {
  formatContactCard,
  parseVcardPhones,
  parseVcardField,
} from './contact-card.js';

// ---------------------------------------------------------------------------
// parseVcardPhones
// ---------------------------------------------------------------------------

test('parseVcardPhones extracts a single TEL field', () => {
  const vcard = 'BEGIN:VCARD\nVERSION:3.0\nFN:Juan\nTEL:+5215512345678\nEND:VCARD';
  const phones = parseVcardPhones(vcard);
  assert.equal(phones.length, 1);
  assert.equal(phones[0].number, '+5215512345678');
  assert.equal(phones[0].type, 'phone');
});

test('parseVcardPhones extracts multiple TEL fields with types', () => {
  const vcard = [
    'BEGIN:VCARD',
    'VERSION:3.0',
    'FN:Taller Ramírez',
    'TEL;type=CELL:+5215512345678',
    'TEL;type=WORK:+5215587654321',
    'TEL;TYPE=HOME,VOICE:+5215500001111',
    'END:VCARD',
  ].join('\n');
  const phones = parseVcardPhones(vcard);
  assert.equal(phones.length, 3);
  assert.equal(phones[0].number, '+5215512345678');
  assert.equal(phones[0].type, 'cell');
  assert.equal(phones[1].number, '+5215587654321');
  assert.equal(phones[1].type, 'work');
  assert.equal(phones[2].number, '+5215500001111');
  assert.equal(phones[2].type, 'home');
});

test('parseVcardPhones handles itemN.TEL prefix', () => {
  const vcard = 'BEGIN:VCARD\nitem1.TEL:+5215512345678\nEND:VCARD';
  const phones = parseVcardPhones(vcard);
  assert.equal(phones.length, 1);
  assert.equal(phones[0].number, '+5215512345678');
});

test('parseVcardPhones returns empty array for null/undefined', () => {
  assert.deepEqual(parseVcardPhones(null), []);
  assert.deepEqual(parseVcardPhones(undefined), []);
  assert.deepEqual(parseVcardPhones(''), []);
});

test('parseVcardPhones returns empty array for vcard with no TEL', () => {
  const vcard = 'BEGIN:VCARD\nVERSION:3.0\nFN:Juan\nEND:VCARD';
  assert.deepEqual(parseVcardPhones(vcard), []);
});

test('parseVcardPhones handles Windows-style line endings', () => {
  const vcard = 'BEGIN:VCARD\r\nTEL:+5215512345678\r\nEND:VCARD';
  const phones = parseVcardPhones(vcard);
  assert.equal(phones.length, 1);
  assert.equal(phones[0].number, '+5215512345678');
});

// ---------------------------------------------------------------------------
// parseVcardField
// ---------------------------------------------------------------------------

test('parseVcardField extracts FN', () => {
  const vcard = 'BEGIN:VCARD\nVERSION:3.0\nFN:Taller Ramírez\nEND:VCARD';
  assert.equal(parseVcardField(vcard, 'FN'), 'Taller Ramírez');
});

test('parseVcardField extracts ORG', () => {
  const vcard = 'BEGIN:VCARD\nORG:AutoServicio Norte S.A.\nEND:VCARD';
  assert.equal(parseVcardField(vcard, 'ORG'), 'AutoServicio Norte S.A.');
});

test('parseVcardField extracts EMAIL', () => {
  const vcard = 'BEGIN:VCARD\nEMAIL;type=WORK:taller@example.com\nEND:VCARD';
  assert.equal(parseVcardField(vcard, 'EMAIL'), 'taller@example.com');
});

test('parseVcardField returns null for missing field', () => {
  const vcard = 'BEGIN:VCARD\nFN:Juan\nEND:VCARD';
  assert.equal(parseVcardField(vcard, 'EMAIL'), null);
});

test('parseVcardField returns null for null input', () => {
  assert.equal(parseVcardField(null, 'FN'), null);
  assert.equal(parseVcardField(undefined, 'FN'), null);
});

test('parseVcardField handles itemN prefix', () => {
  const vcard = 'BEGIN:VCARD\nitem1.EMAIL:test@example.com\nEND:VCARD';
  assert.equal(parseVcardField(vcard, 'EMAIL'), 'test@example.com');
});

// ---------------------------------------------------------------------------
// formatContactCard — single contact (contactMessage)
// ---------------------------------------------------------------------------

test('formatContactCard formats a single contact with name and phone', () => {
  const msg = {
    displayName: 'Taller Ramírez',
    vcard: [
      'BEGIN:VCARD',
      'VERSION:3.0',
      'FN:Taller Ramírez',
      'TEL;type=CELL:+5215587654321',
      'END:VCARD',
    ].join('\n'),
  };
  const result = formatContactCard(msg);
  assert.match(result, /\[contact card\]/);
  assert.match(result, /Name: Taller Ramírez/);
  assert.match(result, /Phone \(cell\): \+5215587654321/);
});

test('formatContactCard extracts org and email from vcard', () => {
  const msg = {
    displayName: 'Juan Pérez',
    vcard: [
      'BEGIN:VCARD',
      'VERSION:3.0',
      'FN:Juan Pérez',
      'ORG:Transportes del Norte',
      'TEL:+5215512345678',
      'EMAIL:juan@transportes.mx',
      'END:VCARD',
    ].join('\n'),
  };
  const result = formatContactCard(msg);
  assert.match(result, /Name: Juan Pérez/);
  assert.match(result, /Org: Transportes del Norte/);
  assert.match(result, /Phone: \+5215512345678/);
  assert.match(result, /Email: juan@transportes.mx/);
});

test('formatContactCard uses vcard FN when displayName is missing', () => {
  const msg = {
    vcard: 'BEGIN:VCARD\nFN:From VCard\nTEL:+123\nEND:VCARD',
  };
  const result = formatContactCard(msg);
  assert.match(result, /Name: From VCard/);
});

// ---------------------------------------------------------------------------
// formatContactCard — multiple contacts (contactsArrayMessage)
// ---------------------------------------------------------------------------

test('formatContactCard formats multiple contacts', () => {
  const msg = {
    contacts: [
      {
        displayName: 'Taller A',
        vcard: 'BEGIN:VCARD\nFN:Taller A\nTEL:+111\nEND:VCARD',
      },
      {
        displayName: 'Taller B',
        vcard: 'BEGIN:VCARD\nFN:Taller B\nTEL:+222\nEND:VCARD',
      },
    ],
  };
  const result = formatContactCard(msg);
  assert.match(result, /\[contact card\]/);
  assert.match(result, /Name: Taller A/);
  assert.match(result, /Phone: \+111/);
  assert.match(result, /---/);  // separator between contacts
  assert.match(result, /Name: Taller B/);
  assert.match(result, /Phone: \+222/);
});

test('formatContactCard skips null entries in contacts array', () => {
  const msg = {
    contacts: [null, { displayName: 'Valid', vcard: 'BEGIN:VCARD\nTEL:+123\nEND:VCARD' }, undefined],
  };
  const result = formatContactCard(msg);
  assert.match(result, /Phone: \+123/);
  assert.doesNotMatch(result, /null/);
});

// ---------------------------------------------------------------------------
// formatContactCard — edge cases
// ---------------------------------------------------------------------------

test('formatContactCard returns empty string for null', () => {
  assert.equal(formatContactCard(null), '');
  assert.equal(formatContactCard(undefined), '');
});

test('formatContactCard returns fallback for empty object', () => {
  assert.equal(formatContactCard({}), '[contact card received]');
});

test('formatContactCard returns fallback for contact with no parseable data', () => {
  const msg = { contacts: [{ randomField: 'abc' }] };
  assert.equal(formatContactCard(msg), '[contact card received]');
});

test('formatContactCard handles contact with multiple phone types', () => {
  const msg = {
    displayName: 'Workshop',
    vcard: [
      'BEGIN:VCARD',
      'FN:Workshop',
      'TEL;type=CELL:+111',
      'TEL;type=WORK:+222',
      'END:VCARD',
    ].join('\n'),
  };
  const result = formatContactCard(msg);
  assert.match(result, /Phone \(cell\): \+111/);
  assert.match(result, /Phone \(work\): \+222/);
});

test('formatContactCard handles real-world WhatsApp vcard format', () => {
  // Actual vCard format from WhatsApp (observed in production)
  const msg = {
    displayName: 'Servicio Automotriz López',
    vcard: [
      'BEGIN:VCARD',
      'VERSION:3.0',
      'N:;Servicio Automotriz López;;;',
      'FN:Servicio Automotriz López',
      'item1.TEL;waid=5215587654321:+52 1 55 8765 4321',
      'item1.X-ABLabel:Celular',
      'END:VCARD',
    ].join('\n'),
  };
  const result = formatContactCard(msg);
  assert.match(result, /Name: Servicio Automotriz López/);
  assert.match(result, /\+52 1 55 8765 4321/);
});
