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
            let pairingHandoffProof = args.pairingHandoffProof;
            const stashedKey = handoffKey;
            handoffKey = null;
            if (pairingHandoffProof === undefined && stashedKey !== null && Date.now() - stashedKey.ts < HANDOFF_KEY_TTL_MS) {
                pairingHandoffProof = hmacSign(companion.prologuePayloadBytes, stashedKey.hmac);
            }
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
                stage: Stage.WaitingForPrimaryIdentity,
                encryptionKey: null,
                verificationCode: null
            };
            opts.logger.debug({ skipHandoffUx }, 'shortcake prologue sent');
            opts.emitPrologueSent?.();
        }
        catch (error) {
            if (activeController === controller) {
                controller.abort();
                activeController = null;
                session = null;
                handoffKey = null;
            }
            throw error;
        }
    };
    /** Derive a handoff HMAC key from the ADV secret and rotate it (matching the official clients). */
    const stashHandoffKeyAndRotateAdv = () => {
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
    /** Build + AES-GCM seal the pairing request and send the `encrypted_pairing_request` IQ. */
    const confirmVerificationCode = async () => {
        if (!session || session.stage !== Stage.WaitingForConfirmation || !session.encryptionKey) {
            throw new Error('shortcake: no verification code awaiting confirmation');
        }
        const current = session;
        current.signal.throwIfAborted();
        const creds = opts.getCreds();
        if (!creds) {
            throw new Error('shortcake: credentials are not initialized');
        }
        const plaintext = proto.PairingRequest.encode({
            companionPublicKey: creds.noiseKey.public,
            companionIdentityKey: creds.signedIdentityKey.public,
            advSecret: Buffer.from(creds.advSecretKey, 'base64')
        }).finish();
        const envelope = encryptPairingRequest(current.encryptionKey, plaintext);
        current.stage = Stage.WaitingForPairing;
        try {
            await opts.query(mdIq('set', [{ tag: 'encrypted_pairing_request', attrs: {}, content: envelope }]));
            current.signal.throwIfAborted();
            session = null;
        }
        catch (error) {
            if (!current.signal.aborted)
                current.stage = Stage.WaitingForConfirmation;
            throw error;
        }
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
        const primaryBytes = getBinaryNodeChildBuffer(node, 'primary_ephemeral_identity');
        const primary = decodePrimaryEphemeralIdentity(primaryBytes);
        await opts.query(mdIq('set', [{ tag: 'companion_nonce', attrs: {}, content: current.companion.companionNonce }]));
        current.signal.throwIfAborted();
        const verificationCode = deriveVerificationCode(current.companion.companionNonce, primary);
        const encryptionKey = deriveEncryptionKey({
            companionPrivKey: current.companion.keyPair.private,
            primaryPublicKey: primary.publicKey,
            deviceType: current.deviceType,
            ref: current.ref
        });
        current.encryptionKey = encryptionKey;
        current.verificationCode = verificationCode;
        current.stage = Stage.WaitingForConfirmation;
        opts.logger.debug('shortcake verification code ready');
        opts.emitVerificationCode?.(verificationCode, current.skipHandoffUx);
        if (current.skipHandoffUx)
            await confirmVerificationCode();
        return true;
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
        confirmVerificationCode,
        hasSession: () => session !== null,
        getVerificationCode: () => session?.verificationCode ?? null,
        clearSession: () => {
            activeController?.abort();
            activeController = null;
            session = null;
            handoffKey = null;
        }
    };
};
