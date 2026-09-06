export class AESECBCipher {
    constructor() {
        this._key = null;
    }

    get algorithm() {
        return { name: "AES-ECB" };
    }

    static async importKey(key, _algorithm, extractable, keyUsages) {
        const cipher = new AESECBCipher;
        await cipher._importKey(key, extractable, keyUsages);
        return cipher;
    }

    async _importKey(key, extractable, keyUsages) {
        this._key = await window.crypto.subtle.importKey(
            "raw", key, {name: "AES-CBC"}, extractable, keyUsages);
    }

    async encrypt(_algorithm, plaintext) {
        const x = new Uint8Array(plaintext);
        if (x.length % 16 !== 0 || this._key === null) {
            return null;
        }
        const n = x.length / 16;
        for (let i = 0; i < n; i++) {
            const y = new Uint8Array(await window.crypto.subtle.encrypt({
                name: "AES-CBC",
                iv: new Uint8Array(16),
            }, this._key, x.slice(i * 16, i * 16 + 16))).slice(0, 16);
            x.set(y, i * 16);
        }
        return x;
    }
}

export class AESEAXCipher {
    constructor() {
        this._rawKey = null;
        this._ctrKey = null;
        this._cbcKey = null;
        this._zeroBlock = new Uint8Array(16);
        this._prefixBlock0 = this._zeroBlock;
        this._prefixBlock1 = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        this._prefixBlock2 = new Uint8Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
    }

    get algorithm() {
        return { name: "AES-EAX" };
    }

    async _encryptBlock(block) {
        const encrypted = await window.crypto.subtle.encrypt({
            name: "AES-CBC",
            iv: this._zeroBlock,
        }, this._cbcKey, block);
        return new Uint8Array(encrypted).slice(0, 16);
    }

    async _initCMAC() {
        const k1 = await this._encryptBlock(this._zeroBlock);
        const k2 = new Uint8Array(16);
        const v = k1[0] >>> 6;
        for (let i = 0; i < 15; i++) {
            k2[i] = (k1[i + 1] >> 6) | (k1[i] << 2);
            k1[i] = (k1[i + 1] >> 7) | (k1[i] << 1);
        }
        const lut = [0x0, 0x87, 0x0e, 0x89];
        k2[14] ^= v >>> 1;
        k2[15] = (k1[15] << 2) ^ lut[v];
        k1[15] = (k1[15] << 1) ^ lut[v >> 1];
        this._k1 = k1;
        this._k2 = k2;
    }

    async _encryptCTR(data, counter) {
        const encrypted = await window.crypto.subtle.encrypt({
            name: "AES-CTR",
            counter: counter,
            length: 128
        }, this._ctrKey, data);
        return new Uint8Array(encrypted);
    }

    async _decryptCTR(data, counter) {
        const decrypted = await window.crypto.subtle.decrypt({
            name: "AES-CTR",
            counter: counter,
            length: 128
        }, this._ctrKey, data);
        return new Uint8Array(decrypted);
    }

    async _computeCMAC(data, prefixBlock) {
        if (prefixBlock.length !== 16) {
            return null;
        }
        const n = Math.floor(data.length / 16);
        const m = Math.ceil(data.length / 16);
        const r = data.length - n * 16;
        const cbcData = new Uint8Array((m + 1) * 16);
        cbcData.set(prefixBlock);
        cbcData.set(data, 16);
        if (r === 0) {
            for (let i = 0; i < 16; i++) {
                cbcData[n * 16 + i] ^= this._k1[i];
            }
        } else {
            cbcData[(n + 1) * 16 + r] = 0x80;
            for (let i = 0; i < 16; i++) {
                cbcData[(n + 1) * 16 + i] ^= this._k2[i];
            }
        }
        let cbcEncrypted = await window.crypto.subtle.encrypt({
            name: "AES-CBC",
            iv: this._zeroBlock,
        }, this._cbcKey, cbcData);

        cbcEncrypted = new Uint8Array(cbcEncrypted);
        const mac = cbcEncrypted.slice(cbcEncrypted.length - 32, cbcEncrypted.length - 16);
        return mac;
    }

    static async importKey(key, _algorithm, _extractable, _keyUsages) {
        const cipher = new AESEAXCipher;
        await cipher._importKey(key);
        return cipher;
    }

    async _importKey(key) {
        this._rawKey = key;
        this._ctrKey = await window.crypto.subtle.importKey(
            "raw", key, {name: "AES-CTR"}, false, ["encrypt", "decrypt"]);
        this._cbcKey = await window.crypto.subtle.importKey(
            "raw", key, {name: "AES-CBC"}, false, ["encrypt"]);
        await this._initCMAC();
    }

    async encrypt(algorithm, message) {
        const ad = algorithm.additionalData;
        const nonce = algorithm.iv;
        const nCMAC = await this._computeCMAC(nonce, this._prefixBlock0);
        const encrypted = await this._encryptCTR(message, nCMAC);
        const adCMAC = await this._computeCMAC(ad, this._prefixBlock1);
        const mac = await this._computeCMAC(encrypted, this._prefixBlock2);
        for (let i = 0; i < 16; i++) {
            mac[i] ^= nCMAC[i] ^ adCMAC[i];
        }
        const res = new Uint8Array(16 + encrypted.length);
        res.set(encrypted);
        res.set(mac, encrypted.length);
        return res;
    }

    async decrypt(algorithm, data) {
        const encrypted = data.slice(0, data.length - 16);
        const ad = algorithm.additionalData;
        const nonce = algorithm.iv;
        const mac = data.slice(data.length - 16);
        const nCMAC = await this._computeCMAC(nonce, this._prefixBlock0);
        const adCMAC = await this._computeCMAC(ad, this._prefixBlock1);
        const computedMac = await this._computeCMAC(encrypted, this._prefixBlock2);
        for (let i = 0; i < 16; i++) {
            computedMac[i] ^= nCMAC[i] ^ adCMAC[i];
        }
        if (computedMac.length !== mac.length) {
            return null;
        }
        for (let i = 0; i < mac.length; i++) {
            if (computedMac[i] !== mac[i]) {
                return null;
            }
        }
        const res = await this._decryptCTR(encrypted, nCMAC);
        return res;
    }
}
