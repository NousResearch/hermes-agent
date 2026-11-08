import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { test } from 'node:test';
import { initAuthCreds, proto, Curve } from '@whiskeysockets/baileys';
import { makeShortcakeFlow } from './passkey_flow.js';
import { attachPasskeyFlow } from './passkey_socket.js';

const logger = { debug() {}, warn() {} };
const assertion = { credentialId: Buffer.from('test-id'), webauthnAssertion: Buffer.from('{}') };
const options = Buffer.from(JSON.stringify({ rpId: 'whatsapp.com', challenge: 'dGVzdA' }));
const request = (from = 's.whatsapp.net') => ({
  tag: 'notification', attrs: { from, type: 'passkey_prologue_request' },
  content: [{ tag: 'passkey_request_options', attrs: {}, content: options }],
});
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
  await new Promise(resolve => setImmediate(resolve));
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
    return { tag: 'iq', attrs: {}, content: [{ tag: 'ref', attrs: {}, content: Buffer.from('ref') }] };
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
  const flow = makeShortcakeFlow({
    logger, getCreds: () => creds, updateCreds: patch => Object.assign(creds, patch),
    signAssertion: async () => assertion,
    query: async node => {
      sent.push(node.content[0].tag);
      return { tag: 'iq', attrs: {}, content: [{ tag: 'ref', attrs: {}, content: Buffer.from('ref') }] };
    },
  });
  await flow.executePrologue({ requestOptions: options });
  await flow.handleIncomingNotification({ tag: 'notification', attrs: { from: 's.whatsapp.net', type: 'crsc_continuation' }, content: [{
    tag: 'primary_ephemeral_identity', attrs: {}, content: proto.PrimaryEphemeralIdentity.encode({
      publicKey: Curve.generateKeyPair().public, nonce: Buffer.alloc(32, 3),
    }).finish(),
  }] });
  assert.equal(sent.includes('encrypted_pairing_request'), false);
  await flow.confirmVerificationCode();
  assert.equal(sent.filter(tag => tag === 'encrypted_pairing_request').length, 1);
  await assert.rejects(flow.confirmVerificationCode(), /no verification code/);
});
