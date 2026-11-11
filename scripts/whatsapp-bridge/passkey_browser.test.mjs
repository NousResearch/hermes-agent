import assert from 'node:assert/strict';
import { createHash, generateKeyPairSync, randomBytes, verify } from 'node:crypto';
import { test } from 'node:test';
import { getEventListeners } from 'node:events';
import { chromium } from 'playwright-core';
import pino from 'pino';
import { proto, initAuthCreds, Curve, hkdf, aesDecryptGCM } from '@whiskeysockets/baileys';
import { makeShortcakeFlow } from './passkey_flow.js';
import { createBrowserPasskeySigner, signPasskeyInPage } from './passkey_browser.js';

const options = () => Buffer.from(JSON.stringify({
  rpId: 'whatsapp.com', challenge: randomBytes(32).toString('base64url'),
  allowCredentials: [], userVerification: 'required', timeout: 5000,
}));

test('rejects a different relying party or page origin before calling WebAuthn', async () => {
  const page = { url: () => 'https://example.com', evaluate: () => assert.fail('must not prompt') };
  await assert.rejects(signPasskeyInPage(page, options(), new AbortController().signal), /official WhatsApp/);
  const wrong = Buffer.from(JSON.stringify({ rpId: 'example.com', challenge: 'abc' }));
  await assert.rejects(signPasskeyInPage(page, wrong, new AbortController().signal), /Unsupported/);
});

test('cancelled socket rejects an outstanding browser prompt', async () => {
  const controller = new AbortController();
  const page = { url: () => 'https://web.whatsapp.com', evaluate: () => new Promise(() => {}) };
  const pending = signPasskeyInPage(page, options(), controller.signal);
  controller.abort();
  await assert.rejects(pending, { name: 'AbortError' });
});

test('cancellation while the browser launches closes it and releases the socket listener', async t => {
  const controller = new AbortController();
  let finishLaunch;
  const launch = new Promise(resolve => { finishLaunch = resolve; });
  t.mock.method(chromium, 'launch', () => launch);
  let closes = 0;
  const pending = createBrowserPasskeySigner('browser')(options(), controller.signal);
  controller.abort();
  finishLaunch({ close: async () => { closes += 1; }, newPage: () => assert.fail('cancelled launch must not navigate') });
  await assert.rejects(pending, { name: 'AbortError' });
  assert.equal(closes, 1);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});

test('browser launch failure releases the socket listener', async t => {
  const controller = new AbortController();
  t.mock.method(chromium, 'launch', async () => { throw new Error('browser unavailable'); });
  await assert.rejects(createBrowserPasskeySigner('missing')(options(), controller.signal), /browser unavailable/);
  assert.equal(getEventListeners(controller.signal, 'abort').length, 0);
});

test('real Chromium WebAuthn assertion completes the salvaged encrypted handshake offline', {
  skip: !process.env.PASSKEY_TEST_BROWSER,
  timeout: 30000,
}, async () => {
  const browser = await chromium.launch({ executablePath: process.env.PASSKEY_TEST_BROWSER, headless: true });
  try {
    const page = await browser.newPage();
    // This route exists only in the test: no WhatsApp account or network request is used.
    await page.route('**/*', route => route.fulfill({ contentType: 'text/html', body: '<title>Offline WebAuthn test</title>' }));
    await page.goto('https://web.whatsapp.com');
    const cdp = await page.context().newCDPSession(page);
    await cdp.send('WebAuthn.enable');
    const { authenticatorId } = await cdp.send('WebAuthn.addVirtualAuthenticator', { options: {
      protocol: 'ctap2', transport: 'internal', hasResidentKey: true,
      hasUserVerification: true, isUserVerified: true, automaticPresenceSimulation: true,
    } });
    const key = generateKeyPairSync('ec', { namedCurve: 'prime256v1' });
    const credentialId = randomBytes(32);
    await cdp.send('WebAuthn.addCredential', { authenticatorId, credential: {
      credentialId: credentialId.toString('base64'), isResidentCredential: true,
      rpId: 'whatsapp.com', privateKey: key.privateKey.export({ type: 'pkcs8', format: 'der' }).toString('base64'),
      userHandle: Buffer.from('offline-owner').toString('base64'), signCount: 0,
    } });
    const requestOptions = options();
    let creds = initAuthCreds();
    const sent = new Map();
    let code;
    let approve;
    const flow = makeShortcakeFlow({
      logger: pino({ level: 'silent' }),
      getCreds: () => creds, updateCreds: patch => { creds = { ...creds, ...patch }; },
      signAssertion: (bytes, signal) => signPasskeyInPage(page, bytes, signal),
      confirmCode: value => { code = value; return new Promise(resolve => { approve = resolve; }); },
      query: async node => {
        const child = node.content[0];
        sent.set(child.tag, child);
        return { tag: 'iq', attrs: { type: 'result' }, content: [{ tag: 'ref', attrs: {}, content: Buffer.from('offline-ref') }] };
      },
    });
    await flow.executePrologue({ requestOptions });
    const prologue = sent.get('passkey_prologue');
    const leaf = name => prologue.content.find(child => child.tag === name).content;
    assert.deepEqual(Buffer.from(leaf('credential_id')), credentialId);
    const assertion = JSON.parse(Buffer.from(leaf('webauthn_assertion')));
    const authData = Buffer.from(assertion.response.authenticatorData, 'base64url');
    const clientData = Buffer.from(assertion.response.clientDataJSON, 'base64url');
    assert.equal(JSON.parse(clientData).challenge, JSON.parse(requestOptions).challenge);
    assert.equal(JSON.parse(clientData).origin, 'https://web.whatsapp.com');
    assert.deepEqual(authData.subarray(0, 32), createHash('sha256').update('whatsapp.com').digest());
    assert.equal(authData[32] & 5, 5, 'user presence and verification flags must be set');
    assert.ok(verify('sha256', Buffer.concat([authData, createHash('sha256').update(clientData).digest()]),
      key.publicKey, Buffer.from(assertion.response.signature, 'base64url')));
    const payload = proto.ProloguePayload.decode(leaf('prologue_payload'));
    const companion = proto.CompanionEphemeralIdentity.decode(payload.companionEphemeralIdentity);
    const primary = Curve.generateKeyPair();
    const continuation = flow.handleIncomingNotification({ tag: 'notification', attrs: { from: 's.whatsapp.net', type: 'crsc_continuation' }, content: [{
      tag: 'primary_ephemeral_identity', attrs: {}, content: proto.PrimaryEphemeralIdentity.encode({ publicKey: primary.public, nonce: randomBytes(32) }).finish(),
    }] });
    await new Promise(resolve => setImmediate(resolve));
    assert.ok(code);
    assert.equal(sent.has('encrypted_pairing_request'), false, 'owner must confirm before pairing data leaves');
    approve();
    await continuation;
    const envelope = proto.EncryptedPairingRequest.decode(sent.get('encrypted_pairing_request').content);
    const secret = Curve.sharedKey(primary.private, companion.publicKey);
    const encryptionKey = hkdf(secret, 32, { salt: Buffer.from(`Companion Pairing ${companion.deviceType} with ref offline-ref`), info: 'Pairing Information Encryption Key' });
    const pairing = proto.PairingRequest.decode(aesDecryptGCM(envelope.encryptedPayload, encryptionKey, envelope.iv, Buffer.alloc(0)));
    assert.deepEqual(Buffer.from(pairing.companionPublicKey), Buffer.from(creds.noiseKey.public));
    assert.deepEqual(Buffer.from(pairing.companionIdentityKey), Buffer.from(creds.signedIdentityKey.public));
    assert.deepEqual(Buffer.from(pairing.advSecret), Buffer.from(creds.advSecretKey, 'base64'));
    assert.equal(flow.hasSession(), false);
  } finally {
    await browser.close();
  }
});
