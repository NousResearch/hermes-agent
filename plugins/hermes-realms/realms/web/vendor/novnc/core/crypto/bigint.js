export function modPow(b, e, m) {
    let r = 1n;
    b = b % m;
    while (e > 0n) {
        if ((e & 1n) === 1n) {
            r = (r * b) % m;
        }
        e = e >> 1n;
        b = (b * b) % m;
    }
    return r;
}

export function bigIntToU8Array(bigint, padLength=0) {
    let hex = bigint.toString(16);
    if (padLength === 0) {
        padLength = Math.ceil(hex.length / 2);
    }
    hex = hex.padStart(padLength * 2, '0');
    const length = hex.length / 2;
    const arr = new Uint8Array(length);
    for (let i = 0; i < length; i++) {
        arr[i] = parseInt(hex.slice(i * 2, i * 2 + 2), 16);
    }
    return arr;
}

export function u8ArrayToBigInt(arr) {
    let hex = '0x';
    for (let i = 0; i < arr.length; i++) {
        hex += arr[i].toString(16).padStart(2, '0');
    }
    return BigInt(hex);
}
