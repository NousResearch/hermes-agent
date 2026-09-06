import Base64 from "../base64.js";
import { modPow, bigIntToU8Array, u8ArrayToBigInt } from "./bigint.js";

export class RSACipher {
    constructor() {
        this._keyLength = 0;
        this._keyBytes = 0;
        this._n = null;
        this._e = null;
        this._d = null;
        this._nBigInt = null;
        this._eBigInt = null;
        this._dBigInt = null;
        this._extractable = false;
    }

    get algorithm() {
        return { name: "RSA-PKCS1-v1_5" };
    }

    _base64urlDecode(data) {
        data = data.replace(/-/g, "+").replace(/_/g, "/");
        data = data.padEnd(Math.ceil(data.length / 4) * 4, "=");
        return Base64.decode(data);
    }

    _padArray(arr, length) {
        const res = new Uint8Array(length);
        res.set(arr, length - arr.length);
        return res;
    }

    static async generateKey(algorithm, extractable, _keyUsages) {
        const cipher = new RSACipher;
        await cipher._generateKey(algorithm, extractable);
        return { privateKey: cipher };
    }

    async _generateKey(algorithm, extractable) {
        this._keyLength = algorithm.modulusLength;
        this._keyBytes = Math.ceil(this._keyLength / 8);
        const key = await window.crypto.subtle.generateKey(
            {
                name: "RSA-OAEP",
                modulusLength: algorithm.modulusLength,
                publicExponent: algorithm.publicExponent,
                hash: {name: "SHA-256"},
            },
            true, ["encrypt", "decrypt"]);
        const privateKey = await window.crypto.subtle.exportKey("jwk", key.privateKey);
        this._n = this._padArray(this._base64urlDecode(privateKey.n), this._keyBytes);
        this._nBigInt = u8ArrayToBigInt(this._n);
        this._e = this._padArray(this._base64urlDecode(privateKey.e), this._keyBytes);
        this._eBigInt = u8ArrayToBigInt(this._e);
        this._d = this._padArray(this._base64urlDecode(privateKey.d), this._keyBytes);
        this._dBigInt = u8ArrayToBigInt(this._d);
        this._extractable = extractable;
    }

    static async importKey(key, _algorithm, extractable, keyUsages) {
        if (keyUsages.length !== 1 || keyUsages[0] !== "encrypt") {
            throw new Error("only support importing RSA public key");
        }
        const cipher = new RSACipher;
        await cipher._importKey(key, extractable);
        return cipher;
    }

    async _importKey(key, extractable) {
        const n = key.n;
        const e = key.e;
        if (n.length !== e.length) {
            throw new Error("the sizes of modulus and public exponent do not match");
        }
        this._keyBytes = n.length;
        this._keyLength = this._keyBytes * 8;
        this._n = new Uint8Array(this._keyBytes);
        this._e = new Uint8Array(this._keyBytes);
        this._n.set(n);
        this._e.set(e);
        this._nBigInt = u8ArrayToBigInt(this._n);
        this._eBigInt = u8ArrayToBigInt(this._e);
        this._extractable = extractable;
    }

    async encrypt(_algorithm, message) {
        if (message.length > this._keyBytes - 11) {
            return null;
        }
        const ps = new Uint8Array(this._keyBytes - message.length - 3);
        window.crypto.getRandomValues(ps);
        for (let i = 0; i < ps.length; i++) {
            ps[i] = Math.floor(ps[i] * 254 / 255 + 1);
        }
        const em = new Uint8Array(this._keyBytes);
        em[1] = 0x02;
        em.set(ps, 2);
        em.set(message, ps.length + 3);
        const emBigInt = u8ArrayToBigInt(em);
        const c = modPow(emBigInt, this._eBigInt, this._nBigInt);
        return bigIntToU8Array(c, this._keyBytes);
    }

    async decrypt(_algorithm, message) {
        if (message.length !== this._keyBytes) {
            return null;
        }
        const msgBigInt = u8ArrayToBigInt(message);
        const emBigInt = modPow(msgBigInt, this._dBigInt, this._nBigInt);
        const em = bigIntToU8Array(emBigInt, this._keyBytes);
        if (em[0] !== 0x00 || em[1] !== 0x02) {
            return null;
        }
        let i = 2;
        for (; i < em.length; i++) {
            if (em[i] === 0x00) {
                break;
            }
        }
        if (i === em.length) {
            return null;
        }
        return em.slice(i + 1, em.length);
    }

    async exportKey() {
        if (!this._extractable) {
            throw new Error("key is not extractable");
        }
        return { n: this._n, e: this._e, d: this._d };
    }
}
