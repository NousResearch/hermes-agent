import { chromium } from 'playwright-core';

const ORIGIN = 'https://web.whatsapp.com';
const MAX_PROMPT_MS = 600_000;

export async function signPasskeyInPage(page, optionsBytes, signal) {
  signal.throwIfAborted();
  const options = JSON.parse(Buffer.from(optionsBytes).toString('utf8'));
  if (options.rpId !== 'whatsapp.com' || typeof options.challenge !== 'string'
      || !/^[A-Za-z0-9_-]+$/.test(options.challenge)) {
    throw new Error('Unsupported WhatsApp passkey request');
  }
  if (new URL(page.url()).origin !== ORIGIN) {
    throw new Error('Passkey verification requires the official WhatsApp Web origin');
  }
  let abort;
  const aborted = new Promise((_, reject) => {
    abort = () => reject(signal.reason);
    signal.addEventListener('abort', abort, { once: true });
  });
  try {
    const result = await Promise.race([
      aborted,
      page.evaluate(async (request) => {
        const decode = value => Uint8Array.from(atob(value.replace(/-/g, '+').replace(/_/g, '/')), c => c.charCodeAt(0));
        const encode = value => {
          let binary = '';
          for (const byte of new Uint8Array(value)) binary += String.fromCharCode(byte);
          return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
        };
        const credential = await navigator.credentials.get({ publicKey: {
          ...request,
          challenge: decode(request.challenge),
          allowCredentials: (request.allowCredentials || []).map(item => ({ ...item, id: decode(item.id) })),
          userVerification: 'required',
        } });
        if (!credential || credential.type !== 'public-key') throw new Error('Passkey verification was cancelled');
        const response = credential.response;
        return {
          id: credential.id, rawId: encode(credential.rawId), type: credential.type,
          response: {
            authenticatorData: encode(response.authenticatorData),
            clientDataJSON: encode(response.clientDataJSON),
            signature: encode(response.signature),
            userHandle: response.userHandle ? encode(response.userHandle) : null,
          },
          clientExtensionResults: credential.getClientExtensionResults(),
        };
      }, options),
    ]);
    signal.throwIfAborted();
    const clientData = JSON.parse(Buffer.from(result.response.clientDataJSON, 'base64url').toString('utf8'));
    if (clientData.type !== 'webauthn.get' || clientData.origin !== ORIGIN || clientData.challenge !== options.challenge) {
      throw new Error('Passkey response does not match the WhatsApp challenge');
    }
    return {
      credentialId: Buffer.from(result.rawId, 'base64url'),
      webauthnAssertion: Buffer.from(JSON.stringify(result)),
    };
  } finally {
    signal.removeEventListener('abort', abort);
  }
}

export function createBrowserPasskeySigner(executablePath) {
  return async (optionsBytes, socketSignal) => {
    socketSignal.throwIfAborted();
    const controller = new AbortController();
    const { signal } = controller;
    const abort = () => controller.abort(socketSignal.reason);
    socketSignal.addEventListener('abort', abort, { once: true });
    const timeout = setTimeout(() => controller.abort(new DOMException('Passkey verification timed out', 'TimeoutError')), MAX_PROMPT_MS);
    timeout.unref();
    let browser;
    const closeOnAbort = () => { void browser.close(); };
    try {
      browser = await chromium.launch({ executablePath, headless: false });
      signal.addEventListener('abort', closeOnAbort, { once: true });
      signal.throwIfAborted();
      const page = await browser.newPage();
      await page.goto(ORIGIN, { waitUntil: 'domcontentloaded', timeout: 60_000 });
      return await signPasskeyInPage(page, optionsBytes, signal);
    } finally {
      clearTimeout(timeout);
      socketSignal.removeEventListener('abort', abort);
      signal.removeEventListener('abort', closeOnAbort);
      if (browser) await browser.close();
    }
  };
}
