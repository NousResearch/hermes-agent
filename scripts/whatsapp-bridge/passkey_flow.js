// Adapted from vinikjkkj's Baileys PR #2689 (MIT). See PASSKEY-LICENSE.
import { randomBytes } from 'node:crypto';
import { proto, getBinaryNodeChild, getBinaryNodeChildBuffer, getBinaryNodeChildString, S_WHATSAPP_NET, hkdf, hmacSign } from '@whiskeysockets/baileys';
import { decodePrimaryEphemeralIdentity, deriveEncryptionKey, deriveVerificationCode, encryptPairingRequest, generateCompanionEphemeralIdentity } from './passkey_crypto.js';
/** HKDF info for the pairing handoff HMAC key (derived from the ADV secret). */
const HANDOFF_KEY_INFO = 'shortcake-passkey-handoff-v1';
const HANDOFF_KEY_TTL_MS = 5 * 60 * 1000;
const Stage = Object.freeze({
    WaitingForPrimaryIdentity: 'waiting_for_primary_identity',
    WaitingForConfirmation: 'waiting_for_confirmation',
    WaitingForPairing: 'waiting_for_pairing'
});
const mdIq = (type, content) => ({
    tag: 'iq',
    attrs: { to: S_WHATSAPP_NET, type, xmlns: 'md' },
    content
});
/** Drives the companion side of the "Shortcake" passkey-linking handshake (md IQ exchange + commit/reveal ECDH). */
export const makeShortcakeFlow = (opts) => {
    let session = null;
    let activeController = null;
    let handoffKey = null;
    const requestPasskeyRequestOptions = async () => {
        const response = await opts.query(mdIq('get', [{ tag: 'passkey_request_options', attrs: {} }]));
        const options = getBinaryNodeChildBuffer(response, 'passkey_request_options');
        if (!options) {
            throw new Error('shortcake: get-passkey-request-options response missing options');
        }
        return options;
    };
    const requestRef = async () => {
        const response = await opts.query(mdIq('get', [{ tag: 'ref', attrs: {} }]));
        const ref = getBinaryNodeChildString(response, 'ref');
        if (!ref) {
            throw new Error('shortcake: get-ref response missing ref');
        }
        return ref;
    };
    /** Fetch options, sign (external), fetch ref, build the commitment, send the `passkey_prologue` IQ. */
    const executePrologue = async (args = {}) => {
        activeController?.abort();
        session = null;
        const controller = new AbortController();
        activeController = controller;
        const { signal } = controller;
        const deviceType = opts.deviceType ?? proto.DeviceProps.PlatformType.CHROME;
        try {
            const requestOptions = args.requestOptions ?? (await requestPasskeyRequestOptions());
            signal.throwIfAborted();
            const assertion = await opts.signAssertion(requestOptions, signal);
            signal.throwIfAborted();
            const ref = await requestRef();
            signal.throwIfAborted();
            const companion = generateCompanionEphemeralIdentity({ ref, deviceType });
            const pairingHandoffProof = handoffKey !== null && Date.now() - handoffKey.ts < HANDOFF_KEY_TTL_MS
                ? hmacSign(companion.prologuePayloadBytes, handoffKey.hmac)
                : undefined;
            const skipHandoffUx = pairingHandoffProof !== undefined;
            const prologueChildren = [
                { tag: 'credential_id', attrs: {}, content: assertion.credentialId },
                { tag: 'webauthn_assertion', attrs: {}, content: assertion.webauthnAssertion },
                { tag: 'prologue_payload', attrs: {}, content: companion.prologuePayloadBytes }
            ];
            if (pairingHandoffProof) {
                prologueChildren.push({ tag: 'pairing_handoff_proof', attrs: {}, content: pairingHandoffProof });
            }
            await opts.query(mdIq('set', [{ tag: 'passkey_prologue', attrs: {}, content: prologueChildren }]));
            signal.throwIfAborted();
            session = {
                signal,
                companion,
                ref,
                deviceType,
                skipHandoffUx,
                stage: Stage.WaitingForPrimaryIdentity
            };
            opts.logger.debug({ skipHandoffUx }, 'shortcake prologue sent');
        }
        catch (error) {
            // Read before the cleanup below aborts it: a signal aborted by a newer prologue or a socket
            // close means the interrupted browser or query call's own error (e.g. ERR_ABORTED) is a
            // cancellation, not a pairing failure.
            const superseded = signal.aborted;
            if (activeController === controller) {
                controller.abort();
                activeController = null;
                session = null;
            }
            throw superseded ? signal.reason : error;
        }
    };
    /** Derive a handoff HMAC key from the ADV secret and rotate it (matching the official clients). */
    const stashHandoffKeyAndRotateAdv = () => {
        // Once per flow: the phone only knows the secret from the QR it scanned, so a superseding request
        // keeps that key (expiring at its use-site window) instead of deriving one from the rotated secret.
        if (handoffKey !== null) {
            return;
        }
        const creds = opts.getCreds();
        if (!creds?.advSecretKey) {
            return;
        }
        handoffKey = {
            hmac: hkdf(Buffer.from(creds.advSecretKey, 'base64'), 32, { info: HANDOFF_KEY_INFO }),
            ts: Date.now()
        };
        opts.updateCreds({ advSecretKey: randomBytes(32).toString('base64') });
    };
    const handlePasskeyPrologueRequest = async (node) => {
        stashHandoffKeyAndRotateAdv();
        const requestOptions = getBinaryNodeChildBuffer(node, 'passkey_request_options');
        opts.logger.debug({ embeddedOptions: !!requestOptions }, 'shortcake prologue requested by server');
        await executePrologue({ requestOptions });
        return true;
    };
    /** AES-GCM seal the pairing request under `encryptionKey` and send the `encrypted_pairing_request` IQ. */
    const sendPairingRequest = async (current, encryptionKey) => {
        current.signal.throwIfAborted();
        const creds = opts.getCreds();
        const plaintext = proto.PairingRequest.encode({
            companionPublicKey: creds.noiseKey.public,
            companionIdentityKey: creds.signedIdentityKey.public,
            advSecret: Buffer.from(creds.advSecretKey, 'base64')
        }).finish();
        const envelope = encryptPairingRequest(encryptionKey, plaintext);
        current.stage = Stage.WaitingForPairing;
        await opts.query(mdIq('set', [{ tag: 'encrypted_pairing_request', attrs: {}, content: envelope }]));
        current.signal.throwIfAborted();
        session = null;
        opts.logger.debug('shortcake encrypted pairing request sent');
    };
    const handlePrimaryEphemeralIdentity = async (node) => {
        const child = getBinaryNodeChild(node, 'primary_ephemeral_identity');
        if (!child) {
            return false;
        }
        if (!session || session.stage !== Stage.WaitingForPrimaryIdentity) {
            opts.logger.warn('shortcake primary identity ignored: no active prologue');
            return true;
        }
        const current = session;
        try {
            await continuePairing(current, decodePrimaryEphemeralIdentity(getBinaryNodeChildBuffer(node, 'primary_ephemeral_identity')));
        }
        catch (error) {
            if (current.signal.aborted)
                throw current.signal.reason;
            // Ctrl+C at the code question rejects with its own AbortError; only this attempt's signal is a supersede.
            throw error?.name === 'AbortError' ? new Error(`Passkey confirmation cancelled: ${error.message}`) : error;
        }
        return true;
    };
    const continuePairing = async (current, primary) => {
        await opts.query(mdIq('set', [{ tag: 'companion_nonce', attrs: {}, content: current.companion.companionNonce }]));
        current.signal.throwIfAborted();
        const verificationCode = deriveVerificationCode(current.companion.companionNonce, primary);
        const encryptionKey = deriveEncryptionKey({
            companionPrivKey: current.companion.keyPair.private,
            primaryPublicKey: primary.publicKey,
            deviceType: current.deviceType,
            ref: current.ref
        });
        current.stage = Stage.WaitingForConfirmation;
        opts.logger.debug({ skipHandoffUx: current.skipHandoffUx }, 'shortcake verification code ready');
        // Without a trusted handoff the owner compares the code with the phone. The attempt's own signal
        // cancels the question when a newer prologue or a socket close replaces this attempt, so a late
        // answer can never approve a different attempt's code.
        if (!current.skipHandoffUx)
            await opts.confirmCode(verificationCode, current.signal);
        await sendPairingRequest(current, encryptionKey);
    };
    /** Routes `passkey_prologue_request` / `crsc_continuation`; returns `true` when consumed. */
    const handleIncomingNotification = async (node) => {
        if (node.attrs.from !== S_WHATSAPP_NET && node.attrs.from !== 's.whatsapp.net')
            return false;
        if (node.attrs.type === 'passkey_prologue_request') {
            return handlePasskeyPrologueRequest(node);
        }
        if (node.attrs.type === 'crsc_continuation') {
            return handlePrimaryEphemeralIdentity(node);
        }
        return false;
    };
    return {
        handleIncomingNotification,
        executePrologue,
        hasSession: () => session !== null,
        clearSession: () => {
            activeController?.abort();
            activeController = null;
            session = null;
        }
    };
};
