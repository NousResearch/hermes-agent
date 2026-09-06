/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2020 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

import { deflateInit, deflate } from "../vendor/pako/lib/zlib/deflate.js";
import { Z_FULL_FLUSH, Z_DEFAULT_COMPRESSION } from "../vendor/pako/lib/zlib/deflate.js";
import ZStream from "../vendor/pako/lib/zlib/zstream.js";

export default class Deflator {
    constructor() {
        this.strm = new ZStream();
        this.chunkSize = 1024 * 10 * 10;
        this.outputBuffer = new Uint8Array(this.chunkSize);

        deflateInit(this.strm, Z_DEFAULT_COMPRESSION);
    }

    deflate(inData) {
        /* eslint-disable camelcase */
        this.strm.input = inData;
        this.strm.avail_in = this.strm.input.length;
        this.strm.next_in = 0;
        this.strm.output = this.outputBuffer;
        this.strm.avail_out = this.chunkSize;
        this.strm.next_out = 0;
        /* eslint-enable camelcase */

        let lastRet = deflate(this.strm, Z_FULL_FLUSH);
        let outData = new Uint8Array(this.strm.output.buffer, 0, this.strm.next_out);

        if (lastRet < 0) {
            throw new Error("zlib deflate failed");
        }

        if (this.strm.avail_in > 0) {
            // Read chunks until done

            let chunks = [outData];
            let totalLen = outData.length;
            do {
                /* eslint-disable camelcase */
                this.strm.output = new Uint8Array(this.chunkSize);
                this.strm.next_out = 0;
                this.strm.avail_out = this.chunkSize;
                /* eslint-enable camelcase */

                lastRet = deflate(this.strm, Z_FULL_FLUSH);

                if (lastRet < 0) {
                    throw new Error("zlib deflate failed");
                }

                let chunk = new Uint8Array(this.strm.output.buffer, 0, this.strm.next_out);
                totalLen += chunk.length;
                chunks.push(chunk);
            } while (this.strm.avail_in > 0);

            // Combine chunks into a single data

            let newData = new Uint8Array(totalLen);
            let offset = 0;

            for (let i = 0; i < chunks.length; i++) {
                newData.set(chunks[i], offset);
                offset += chunks[i].length;
            }

            outData = newData;
        }

        /* eslint-disable camelcase */
        this.strm.input = null;
        this.strm.avail_in = 0;
        this.strm.next_in = 0;
        /* eslint-enable camelcase */

        return outData;
    }

}
