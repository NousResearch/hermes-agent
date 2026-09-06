/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2021 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

/*
 * Performs MD5 hashing on an array of bytes, returns an array of bytes
 */

export async function MD5(d) {
    let s = "";
    for (let i = 0; i < d.length; i++) {
        s += String.fromCharCode(d[i]);
    }
    return M(V(Y(X(s), 8 * s.length)));
}

function M(d) {
    let f = new Uint8Array(d.length);
    for (let i=0;i<d.length;i++) {
        f[i] = d.charCodeAt(i);
    }
    return f;
}

function X(d) {
    let r = Array(d.length >> 2);
    for (let m = 0; m < r.length; m++) r[m] = 0;
    for (let m = 0; m < 8 * d.length; m += 8) r[m >> 5] |= (255 & d.charCodeAt(m / 8)) << m % 32;
    return r;
}

function V(d) {
    let r = "";
    for (let m = 0; m < 32 * d.length; m += 8) r += String.fromCharCode(d[m >> 5] >>> m % 32 & 255);
    return r;
}

function Y(d, g) {
    d[g >> 5] |= 128 << g % 32, d[14 + (g + 64 >>> 9 << 4)] = g;
    let m = 1732584193, f = -271733879, r = -1732584194, i = 271733878;
    for (let n = 0; n < d.length; n += 16) {
        let h = m,
            t = f,
            g = r,
            e = i;
        f = ii(f = ii(f = ii(f = ii(f = hh(f = hh(f = hh(f = hh(f = gg(f = gg(f = gg(f = gg(f = ff(f = ff(f = ff(f = ff(f, r = ff(r, i = ff(i, m = ff(m, f, r, i, d[n + 0], 7, -680876936), f, r, d[n + 1], 12, -389564586), m, f, d[n + 2], 17, 606105819), i, m, d[n + 3], 22, -1044525330), r = ff(r, i = ff(i, m = ff(m, f, r, i, d[n + 4], 7, -176418897), f, r, d[n + 5], 12, 1200080426), m, f, d[n + 6], 17, -1473231341), i, m, d[n + 7], 22, -45705983), r = ff(r, i = ff(i, m = ff(m, f, r, i, d[n + 8], 7, 1770035416), f, r, d[n + 9], 12, -1958414417), m, f, d[n + 10], 17, -42063), i, m, d[n + 11], 22, -1990404162), r = ff(r, i = ff(i, m = ff(m, f, r, i, d[n + 12], 7, 1804603682), f, r, d[n + 13], 12, -40341101), m, f, d[n + 14], 17, -1502002290), i, m, d[n + 15], 22, 1236535329), r = gg(r, i = gg(i, m = gg(m, f, r, i, d[n + 1], 5, -165796510), f, r, d[n + 6], 9, -1069501632), m, f, d[n + 11], 14, 643717713), i, m, d[n + 0], 20, -373897302), r = gg(r, i = gg(i, m = gg(m, f, r, i, d[n + 5], 5, -701558691), f, r, d[n + 10], 9, 38016083), m, f, d[n + 15], 14, -660478335), i, m, d[n + 4], 20, -405537848), r = gg(r, i = gg(i, m = gg(m, f, r, i, d[n + 9], 5, 568446438), f, r, d[n + 14], 9, -1019803690), m, f, d[n + 3], 14, -187363961), i, m, d[n + 8], 20, 1163531501), r = gg(r, i = gg(i, m = gg(m, f, r, i, d[n + 13], 5, -1444681467), f, r, d[n + 2], 9, -51403784), m, f, d[n + 7], 14, 1735328473), i, m, d[n + 12], 20, -1926607734), r = hh(r, i = hh(i, m = hh(m, f, r, i, d[n + 5], 4, -378558), f, r, d[n + 8], 11, -2022574463), m, f, d[n + 11], 16, 1839030562), i, m, d[n + 14], 23, -35309556), r = hh(r, i = hh(i, m = hh(m, f, r, i, d[n + 1], 4, -1530992060), f, r, d[n + 4], 11, 1272893353), m, f, d[n + 7], 16, -155497632), i, m, d[n + 10], 23, -1094730640), r = hh(r, i = hh(i, m = hh(m, f, r, i, d[n + 13], 4, 681279174), f, r, d[n + 0], 11, -358537222), m, f, d[n + 3], 16, -722521979), i, m, d[n + 6], 23, 76029189), r = hh(r, i = hh(i, m = hh(m, f, r, i, d[n + 9], 4, -640364487), f, r, d[n + 12], 11, -421815835), m, f, d[n + 15], 16, 530742520), i, m, d[n + 2], 23, -995338651), r = ii(r, i = ii(i, m = ii(m, f, r, i, d[n + 0], 6, -198630844), f, r, d[n + 7], 10, 1126891415), m, f, d[n + 14], 15, -1416354905), i, m, d[n + 5], 21, -57434055), r = ii(r, i = ii(i, m = ii(m, f, r, i, d[n + 12], 6, 1700485571), f, r, d[n + 3], 10, -1894986606), m, f, d[n + 10], 15, -1051523), i, m, d[n + 1], 21, -2054922799), r = ii(r, i = ii(i, m = ii(m, f, r, i, d[n + 8], 6, 1873313359), f, r, d[n + 15], 10, -30611744), m, f, d[n + 6], 15, -1560198380), i, m, d[n + 13], 21, 1309151649), r = ii(r, i = ii(i, m = ii(m, f, r, i, d[n + 4], 6, -145523070), f, r, d[n + 11], 10, -1120210379), m, f, d[n + 2], 15, 718787259), i, m, d[n + 9], 21, -343485551), m = add(m, h), f = add(f, t), r = add(r, g), i = add(i, e);
    }
    return Array(m, f, r, i);
}

function cmn(d, g, m, f, r, i) {
    return add(rol(add(add(g, d), add(f, i)), r), m);
}

function ff(d, g, m, f, r, i, n) {
    return cmn(g & m | ~g & f, d, g, r, i, n);
}

function gg(d, g, m, f, r, i, n) {
    return cmn(g & f | m & ~f, d, g, r, i, n);
}

function hh(d, g, m, f, r, i, n) {
    return cmn(g ^ m ^ f, d, g, r, i, n);
}

function ii(d, g, m, f, r, i, n) {
    return cmn(m ^ (g | ~f), d, g, r, i, n);
}

function add(d, g) {
    let m = (65535 & d) + (65535 & g);
    return (d >> 16) + (g >> 16) + (m >> 16) << 16 | 65535 & m;
}

function rol(d, g) {
    return d << g | d >>> 32 - g;
}
