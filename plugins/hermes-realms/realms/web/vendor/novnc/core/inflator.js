/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2020 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

import { inflateInit, inflate, inflateReset } from "../vendor/pako/lib/zlib/inflate.js";
import ZStream from "../vendor/pako/lib/zlib/zstream.js";

export default class Inflate {
    constructor() {
        this.strm = new ZStream();
        this.chunkSize = 1024 * 10 * 10;
        this.strm.output = new Uint8Array(this.chunkSize);

        inflateInit(this.strm);
    }

    setInput(data) {
        if (!data) {
            //FIXME: flush remaining data.
            /* eslint-disable camelcase */
            this.strm.input = null;
            this.strm.avail_in = 0;
            this.strm.next_in = 0;
        } else {
            this.strm.input = data;
            this.strm.avail_in = this.strm.input.length;
            this.strm.next_in = 0;
            /* eslint-enable camelcase */
        }
    }

    inflate(expected) {
        // resize our output buffer if it's too small
        // (we could just use multiple chunks, but that would cause an extra
        // allocation each time to flatten the chunks)
        if (expected > this.chunkSize) {
            this.chunkSize = expected;
            this.strm.output = new Uint8Array(this.chunkSize);
        }

        /* eslint-disable camelcase */
        this.strm.next_out = 0;
        this.strm.avail_out = expected;
        /* eslint-enable camelcase */

        let ret = inflate(this.strm, 0); // Flush argument not used.
        if (ret < 0) {
            throw new Error("zlib inflate failed");
        }

        if (this.strm.next_out != expected) {
            throw new Error("Incomplete zlib block");
        }

        return new Uint8Array(this.strm.output.buffer, 0, this.strm.next_out);
    }

    reset() {
        inflateReset(this.strm);
    }
}
