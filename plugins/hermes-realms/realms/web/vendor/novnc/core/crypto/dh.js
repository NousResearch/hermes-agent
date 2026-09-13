import { modPow, bigIntToU8Array, u8ArrayToBigInt } from "./bigint.js";

class DHPublicKey {
    constructor(key) {
        this._key = key;
    }

    get algorithm() {
        return { name: "DH" };
    }

    exportKey() {
        return this._key;
    }
}

export class DHCipher {
    constructor() {
        this._g = null;
        this._p = null;
        this._gBigInt = null;
        this._pBigInt = null;
        this._privateKey = null;
    }

    get algorithm() {
        return { name: "DH" };
    }

    static generateKey(algorithm, _extractable) {
        const cipher = new DHCipher;
        cipher._generateKey(algorithm);
        return { privateKey: cipher, publicKey: new DHPublicKey(cipher._publicKey) };
    }

    _generateKey(algorithm) {
        const g = algorithm.g;
        const p = algorithm.p;
        this._keyBytes = p.length;
        this._gBigInt = u8ArrayToBigInt(g);
        this._pBigInt = u8ArrayToBigInt(p);
        this._privateKey = window.crypto.getRandomValues(new Uint8Array(this._keyBytes));
        this._privateKeyBigInt = u8ArrayToBigInt(this._privateKey);
        this._publicKey = bigIntToU8Array(modPow(
            this._gBigInt, this._privateKeyBigInt, this._pBigInt), this._keyBytes);
    }

    deriveBits(algorithm, length) {
        const bytes = Math.ceil(length / 8);
        const pkey = new Uint8Array(algorithm.public);
        const len = bytes > this._keyBytes ? bytes : this._keyBytes;
        const secret = modPow(u8ArrayToBigInt(pkey), this._privateKeyBigInt, this._pBigInt);
        return bigIntToU8Array(secret, len).slice(0, len);
    }
}
