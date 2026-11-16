import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';
import { initAuthCreds, proto, Curve, hkdf, hmacSign } from '@whiskeysockets/baileys';
import { makeShortcakeFlow } from './passkey_flow.js';
import { attachPasskeyFlow, loadPasskeySigner } from './passkey_socket.js';

test('invalid pairing modes reject before loading the browser dependency', async () => {
  const load = () => assert.fail('invalid mode must not import Playwright');
  for (const mode of [
    { pairOnly: false, pairJson: false, isTTY: true },
    { pairOnly: true, pairJson: true, isTTY: true },
    { pairOnly: true, pairJson: false, isTTY: false },
  ]) {
    await assert.rejects(loadPasskeySigner('browser', mode, load), /requires interactive --pair-only/);
  }
  assert.equal(await loadPasskeySigner('', {}, load), undefined);
  const signer = () => {};
  assert.equal(await loadPasskeySigner('browser', { pairOnly: true, pairJson: false, isTTY: true }, async () => ({
    createBrowserPasskeySigner: executable => { assert.equal(executable, 'browser'); return signer; },
  })), signer);
});

const logger = { debug() {}, warn() {} };
const assertion = { credentialId: Buffer.from('test-id'), webauthnAssertion: Buffer.from('{}') };
const options = Buffer.from(JSON.stringify({ rpId: 'whatsapp.com', challenge: 'dGVzdA' }));
const request = (from = 's.whatsapp.net') => ({
  tag: 'notification', attrs: { from, type: 'passkey_prologue_request' },
  content: [{ tag: 'passkey_request_options', attrs: {}, content: options }],
});
const refReply = () => ({ tag: 'iq', attrs: {}, content: [{ tag: 'ref', attrs: {}, content: Buffer.from('ref') }] });
const continuation = () => ({ tag: 'notification', attrs: { from: 's.whatsapp.net', type: 'crsc_continuation' }, content: [{
  tag: 'primary_ephemeral_identity', attrs: {}, content: proto.PrimaryEphemeralIdentity.encode({
    publicKey: Curve.generateKeyPair().public, nonce: Buffer.alloc(32, 3),
  }).finish(),
}] });
const settle = () => new Promise(resolve => setImmediate(resolve));

function socket() {
  const ws = new EventEmitter();
  const ev = new EventEmitter();
  const creds = initAuthCreds();
  ev.on('creds.update', patch => Object.assign(creds, patch));
  return { ws, ev, authState: { creds } };
}

test('official socket event contract cancels queued and active signing on close', async () => {
  const sock = socket();
  let finish;
  let signal;
  let started;
  const signingStarted = new Promise(resolve => { started = resolve; });
  const signing = new Promise(resolve => { finish = resolve; });
  let calls = 0;
  sock.query = async () => assert.fail('closed socket must not send a prologue');
  attachPasskeyFlow(sock, { logger, signAssertion: async (_bytes, abortSignal) => {
    calls += 1;
    signal = abortSignal;
    started();
    return signing;
  } });
  sock.ws.emit('CB:notification', request());
  await signingStarted;
  sock.ws.emit('CB:notification', request());
  sock.ev.emit('connection.update', { connection: 'close' });
  assert.equal(signal.aborted, true);
  assert.equal(sock.ws.listenerCount('CB:notification'), 0);
  finish(assertion);
  await settle();
  assert.equal(calls, 1);
});

test('missing signer is reported and another account cannot trigger a prompt', () => {
  const sock = socket();
  const events = [];
  sock.ev.on('connection.update', update => events.push(update));
  attachPasskeyFlow(sock, { logger });
  const before = sock.authState.creds.advSecretKey;
  sock.ws.emit('CB:notification', request('15550000000@s.whatsapp.net'));
  assert.deepEqual(events, []);
  assert.equal(sock.authState.creds.advSecretKey, before);
  sock.ws.emit('CB:notification', request());
  assert.deepEqual(events, [{ passkeyRequired: { hasSigner: false } }]);
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('signer failure surfaces through the socket and a fresh attempt can continue', async () => {
  const sock = socket();
  let calls = 0;
  let prologueSent;
  const prologue = new Promise(resolve => { prologueSent = resolve; });
  sock.query = async node => {
    if (node.content[0].tag === 'passkey_prologue') prologueSent();
    return refReply();
  };
  attachPasskeyFlow(sock, { logger, signAssertion: async () => {
    calls += 1;
    if (calls === 1) throw new Error('owner declined');
    return assertion;
  } });
  const failure = new Promise(resolve => sock.ev.on('connection.update', update => {
    if (update.passkeyError) resolve(update.passkeyError);
  }));
  sock.ws.emit('CB:notification', request());
  assert.equal((await failure).message, 'owner declined');
  sock.ws.emit('CB:notification', request());
  await prologue;
  assert.equal(calls, 2);
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('pairing payload waits for owner confirmation without trusted handoff', async () => {
  const creds = initAuthCreds();
  const sent = [];
  let approve;
  let asked;
  const question = new Promise(resolve => { asked = resolve; });
  const flow = makeShortcakeFlow({
    logger, getCreds: () => creds, updateCreds: patch => Object.assign(creds, patch),
    signAssertion: async () => assertion,
    confirmCode: () => new Promise(resolve => { approve = resolve; asked(); }),
    query: async node => { sent.push(node.content[0].tag); return refReply(); },
  });
  await flow.executePrologue({ requestOptions: options });
  const handled = flow.handleIncomingNotification(continuation());
  await question;
  assert.equal(sent.includes('encrypted_pairing_request'), false);
  approve();
  await handled;
  assert.equal(sent.filter(tag => tag === 'encrypted_pairing_request').length, 1);
  assert.equal(flow.hasSession(), false);
});

// Production entry point: the server's prologue request goes through CB:notification, which stashes the
// ADV-derived handoff key first. Pin both outcomes so the confirmation prompt's reachability is explicit.
async function driveServerPrologue(t, { signingMs, confirmCode }) {
  const sock = socket();
  const sent = [];
  let clock = 1_000_000;
  t.mock.method(Date, 'now', () => clock);
  let prologueSent;
  const prologue = new Promise(resolve => { prologueSent = resolve; });
  sock.query = async node => {
    sent.push(node.content[0].tag);
    if (node.content[0].tag === 'passkey_prologue') setImmediate(prologueSent);
    return refReply();
  };
  attachPasskeyFlow(sock, { logger, confirmCode, signAssertion: async () => { clock += signingMs; return assertion; } });
  sock.ws.emit('CB:notification', request());
  await prologue;
  sock.ws.emit('CB:notification', continuation());
  await settle();
  return { sock, sent };
}

test('server prologue within the handoff window sends the pairing request without a prompt', async t => {
  const confirmCode = t.mock.fn(async () => {});
  const { sock, sent } = await driveServerPrologue(t, { signingMs: 1000, confirmCode });
  assert.equal(confirmCode.mock.callCount(), 0);
  assert.deepEqual(sent, ['ref', 'passkey_prologue', 'companion_nonce', 'encrypted_pairing_request']);
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('server prologue past the handoff window waits for owner confirmation', async t => {
  let approve;
  const confirmCode = t.mock.fn(() => new Promise(resolve => { approve = resolve; }));
  const { sock, sent } = await driveServerPrologue(t, { signingMs: 6 * 60 * 1000, confirmCode });
  assert.equal(confirmCode.mock.callCount(), 1);
  assert.match(confirmCode.mock.calls[0].arguments[0], /^[0-9A-Z]{8}$/);
  assert.equal(sent.includes('encrypted_pairing_request'), false);
  approve();
  await settle();
  assert.equal(sent.at(-1), 'encrypted_pairing_request');
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('a newer prologue cancels the pending code question and a late answer sends nothing', async t => {
  const questions = [];
  // Ignores its signal on purpose, like a prompt that still returns "y" after being superseded.
  const confirmCode = (code, signal) => new Promise(resolve => { questions.push({ code, signal, resolve }); });
  const { sock, sent } = await driveServerPrologue(t, { signingMs: 6 * 60 * 1000, confirmCode });
  const errors = [];
  sock.ev.on('connection.update', update => { if (update.passkeyError) errors.push(update.passkeyError); });
  assert.equal(questions.length, 1);
  sock.ws.emit('CB:notification', request());
  assert.equal(questions[0].signal.aborted, true, 'the superseded question must be cancelled at once');
  questions[0].resolve();
  await settle();
  assert.equal(sent.includes('encrypted_pairing_request'), false);
  assert.deepEqual(errors, [], 'a superseded attempt is a cancellation, not a pairing failure');
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('a code question that fails with a plain error after supersede is a cancellation', async t => {
  // A closed readline interface rejects with its own error rather than the abort reason.
  const confirmCode = (_code, signal) => new Promise((_, reject) => {
    signal.addEventListener('abort', () => reject(new Error('readline was closed')), { once: true });
  });
  const { sock, sent } = await driveServerPrologue(t, { signingMs: 6 * 60 * 1000, confirmCode });
  const errors = [];
  sock.ev.on('connection.update', update => { if (update.passkeyError) errors.push(update.passkeyError); });
  sock.ws.emit('CB:notification', request());
  await settle();
  assert.deepEqual(errors, []);
  assert.equal(sent.includes('encrypted_pairing_request'), false);
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('Ctrl+C at the code question ends the attempt instead of passing as a supersede', async t => {
  let interrupt;
  const confirmCode = () => new Promise((_, reject) => { interrupt = reject; });
  const { sock, sent } = await driveServerPrologue(t, { signingMs: 6 * 60 * 1000, confirmCode });
  const errors = [];
  sock.ev.on('connection.update', update => { if (update.passkeyError) errors.push(update.passkeyError); });
  // What readline/promises question() rejects with on Ctrl+C: an AbortError the attempt's signal never sent.
  interrupt(Object.assign(new Error('Aborted with Ctrl+C'), { name: 'AbortError', code: 'ABORT_ERR' }));
  await settle();
  assert.equal(errors.length, 1, 'the bridge must report the cancelled prompt and exit');
  assert.match(errors[0].message, /cancelled.*Ctrl\+C/);
  assert.equal(sent.includes('encrypted_pairing_request'), false);
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('a superseding prologue request proves the handoff with the secret from the scanned QR', async () => {
  const sock = socket();
  const qrSecret = sock.authState.creds.advSecretKey;
  const prologues = [];
  sock.query = async node => {
    if (node.content[0].tag === 'passkey_prologue') prologues.push(node.content[0].content);
    return refReply();
  };
  const signals = [];
  attachPasskeyFlow(sock, { logger, confirmCode: async () => {}, signAssertion: (_bytes, signal) => {
    signals.push(signal);
    if (signals.length > 1) return Promise.resolve(assertion);
    return new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason), { once: true }));
  } });
  sock.ws.emit('CB:notification', request());
  await settle();
  sock.ws.emit('CB:notification', request());
  await settle();
  await settle();
  assert.equal(prologues.length, 1);
  assert.notEqual(sock.authState.creds.advSecretKey, qrSecret, 'the first request still rotates the ADV secret');
  const leaf = name => prologues[0].find(child => child.tag === name)?.content;
  const key = hkdf(Buffer.from(qrSecret, 'base64'), 32, { info: 'shortcake-passkey-handoff-v1' });
  assert.deepEqual(Buffer.from(leaf('pairing_handoff_proof')), Buffer.from(hmacSign(leaf('prologue_payload'), key)));
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('a request after the first prologue was sent still proves the handoff with the QR secret', async () => {
  const sock = socket();
  const qrSecret = sock.authState.creds.advSecretKey;
  const prologues = [];
  sock.query = async node => {
    if (node.content[0].tag === 'passkey_prologue') prologues.push(node.content[0].content);
    return refReply();
  };
  attachPasskeyFlow(sock, { logger, confirmCode: async () => {}, signAssertion: async () => assertion });
  sock.ws.emit('CB:notification', request());
  await settle();
  sock.ws.emit('CB:notification', request());
  await settle();
  assert.equal(prologues.length, 2);
  const key = hkdf(Buffer.from(qrSecret, 'base64'), 32, { info: 'shortcake-passkey-handoff-v1' });
  for (const prologue of prologues) {
    const leaf = name => prologue.find(child => child.tag === name)?.content;
    assert.deepEqual(Buffer.from(leaf('pairing_handoff_proof')), Buffer.from(hmacSign(leaf('prologue_payload'), key)));
  }
  sock.ev.emit('connection.update', { connection: 'close' });
});

test('superseding an attempt whose browser call fails with a plain error keeps the bridge pairing', async () => {
  const sock = socket();
  const errors = [];
  sock.ev.on('connection.update', update => { if (update.passkeyError) errors.push(update.passkeyError); });
  const signals = [];
  let prologues = 0;
  sock.query = async node => { if (node.content[0].tag === 'passkey_prologue') prologues += 1; return refReply(); };
  attachPasskeyFlow(sock, { logger, confirmCode: async () => {}, signAssertion: (_bytes, signal) => {
    signals.push(signal);
    if (signals.length > 1) return Promise.resolve(assertion);
    // Playwright rejects an interrupted page.goto with a plain Error once the browser closes.
    return new Promise((_, reject) => signal.addEventListener('abort', () => reject(new Error('page.goto: net::ERR_ABORTED')), { once: true }));
  } });
  sock.ws.emit('CB:notification', request());
  await settle();
  sock.ws.emit('CB:notification', request());
  await settle();
  await settle();
  assert.equal(signals.length, 2);
  assert.deepEqual(errors, []);
  assert.equal(prologues, 1, 'the replacement attempt sends its prologue');
  sock.ev.emit('connection.update', { connection: 'close' });
});
