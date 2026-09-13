import { AESECBCipher, AESEAXCipher } from "./aes.js";
import { DESCBCCipher, DESECBCipher } from "./des.js";
import { RSACipher } from "./rsa.js";
import { DHCipher } from "./dh.js";
import { MD5 } from "./md5.js";

// A single interface for the cryptographic algorithms not supported by SubtleCrypto.
// Both synchronous and asynchronous implmentations are allowed.
class LegacyCrypto {
    constructor() {
        this._algorithms = {
            "AES-ECB": AESECBCipher,
            "AES-EAX": AESEAXCipher,
            "DES-ECB": DESECBCipher,
            "DES-CBC": DESCBCCipher,
            "RSA-PKCS1-v1_5": RSACipher,
            "DH": DHCipher,
            "MD5": MD5,
        };
    }

    encrypt(algorithm, key, data) {
        if (key.algorithm.name !== algorithm.name) {
            throw new Error("algorithm does not match");
        }
        if (typeof key.encrypt !== "function") {
            throw new Error("key does not support encryption");
        }
        return key.encrypt(algorithm, data);
    }

    decrypt(algorithm, key, data) {
        if (key.algorithm.name !== algorithm.name) {
            throw new Error("algorithm does not match");
        }
        if (typeof key.decrypt !== "function") {
            throw new Error("key does not support encryption");
        }
        return key.decrypt(algorithm, data);
    }

    importKey(format, keyData, algorithm, extractable, keyUsages) {
        if (format !== "raw") {
            throw new Error("key format is not supported");
        }
        const alg = this._algorithms[algorithm.name];
        if (typeof alg === "undefined" || typeof alg.importKey !== "function") {
            throw new Error("algorithm is not supported");
        }
        return alg.importKey(keyData, algorithm, extractable, keyUsages);
    }

    generateKey(algorithm, extractable, keyUsages) {
        const alg = this._algorithms[algorithm.name];
        if (typeof alg === "undefined" || typeof alg.generateKey !== "function") {
            throw new Error("algorithm is not supported");
        }
        return alg.generateKey(algorithm, extractable, keyUsages);
    }

    exportKey(format, key) {
        if (format !== "raw") {
            throw new Error("key format is not supported");
        }
        if (typeof key.exportKey !== "function") {
            throw new Error("key does not support exportKey");
        }
        return key.exportKey();
    }

    digest(algorithm, data) {
        const alg = this._algorithms[algorithm];
        if (typeof alg !== "function") {
            throw new Error("algorithm is not supported");
        }
        return alg(data);
    }

    deriveBits(algorithm, key, length) {
        if (key.algorithm.name !== algorithm.name) {
            throw new Error("algorithm does not match");
        }
        if (typeof key.deriveBits !== "function") {
            throw new Error("key does not support deriveBits");
        }
        return key.deriveBits(algorithm, length);
    }
}

export default new LegacyCrypto;
