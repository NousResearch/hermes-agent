/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2024 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 *
 */

import Inflator from "../inflator.js";

export default class ZlibDecoder {
    constructor() {
        this._zlib = new Inflator();
        this._length = 0;
    }

    decodeRect(x, y, width, height, sock, display, depth) {
        if ((width === 0) || (height === 0)) {
            return true;
        }

        if (this._length === 0) {
            if (sock.rQwait("ZLIB", 4)) {
                return false;
            }

            this._length = sock.rQshift32();
        }

        if (sock.rQwait("ZLIB", this._length)) {
            return false;
        }

        let data = new Uint8Array(sock.rQshiftBytes(this._length, false));
        this._length = 0;

        this._zlib.setInput(data);
        data = this._zlib.inflate(width * height * 4);
        this._zlib.setInput(null);

        // Max sure the image is fully opaque
        for (let i = 0; i < width * height; i++) {
            data[i * 4 + 3] = 255;
        }

        display.blitImage(x, y, width, height, data, 0);

        return true;
    }
}
