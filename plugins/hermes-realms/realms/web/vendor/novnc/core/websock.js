/*
 * Websock: high-performance buffering wrapper
 * Copyright (C) 2019 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * Websock is similar to the standard WebSocket / RTCDataChannel object
 * but with extra buffer handling.
 *
 * Websock has built-in receive queue buffering; the message event
 * does not contain actual data but is simply a notification that
 * there is new data available. Several rQ* methods are available to
 * read binary data off of the receive queue.
 */

import * as Log from './util/logging.js';

// this has performance issues in some versions Chromium, and
// doesn't gain a tremendous amount of performance increase in Firefox
// at the moment.  It may be valuable to turn it on in the future.
const MAX_RQ_GROW_SIZE = 40 * 1024 * 1024;  // 40 MiB

// Constants pulled from RTCDataChannelState enum
// https://developer.mozilla.org/en-US/docs/Web/API/RTCDataChannel/readyState#RTCDataChannelState_enum
const DataChannel = {
    CONNECTING: "connecting",
    OPEN: "open",
    CLOSING: "closing",
    CLOSED: "closed"
};

const ReadyStates = {
    CONNECTING: [WebSocket.CONNECTING, DataChannel.CONNECTING],
    OPEN: [WebSocket.OPEN, DataChannel.OPEN],
    CLOSING: [WebSocket.CLOSING, DataChannel.CLOSING],
    CLOSED: [WebSocket.CLOSED, DataChannel.CLOSED],
};

// Properties a raw channel must have, WebSocket and RTCDataChannel are two examples
const rawChannelProps = [
    "send",
    "close",
    "binaryType",
    "onerror",
    "onmessage",
    "onopen",
    "protocol",
    "readyState",
];

export default class Websock {
    constructor() {
        this._websocket = null;  // WebSocket or RTCDataChannel object

        this._rQi = 0;           // Receive queue index
        this._rQlen = 0;         // Next write position in the receive queue
        this._rQbufferSize = 1024 * 1024 * 4; // Receive queue buffer size (4 MiB)
        // called in init: this._rQ = new Uint8Array(this._rQbufferSize);
        this._rQ = null; // Receive queue

        this._sQbufferSize = 1024 * 10;  // 10 KiB
        // called in init: this._sQ = new Uint8Array(this._sQbufferSize);
        this._sQlen = 0;
        this._sQ = null;  // Send queue

        this._eventHandlers = {
            message: () => {},
            open: () => {},
            close: () => {},
            error: () => {}
        };
    }

    // Getters and setters

    get readyState() {
        let subState;

        if (this._websocket === null) {
            return "unused";
        }

        subState = this._websocket.readyState;

        if (ReadyStates.CONNECTING.includes(subState)) {
            return "connecting";
        } else if (ReadyStates.OPEN.includes(subState)) {
            return "open";
        } else if (ReadyStates.CLOSING.includes(subState)) {
            return "closing";
        } else if (ReadyStates.CLOSED.includes(subState)) {
            return "closed";
        }

        return "unknown";
    }

    // Receive queue
    rQpeek8() {
        return this._rQ[this._rQi];
    }

    rQskipBytes(bytes) {
        this._rQi += bytes;
    }

    rQshift8() {
        return this._rQshift(1);
    }

    rQshift16() {
        return this._rQshift(2);
    }

    rQshift32() {
        return this._rQshift(4);
    }

    // TODO(directxman12): test performance with these vs a DataView
    _rQshift(bytes) {
        let res = 0;
        for (let byte = bytes - 1; byte >= 0; byte--) {
            res += this._rQ[this._rQi++] << (byte * 8);
        }
        return res >>> 0;
    }

    rQlen() {
        return this._rQlen - this._rQi;
    }

    rQshiftStr(len) {
        let str = "";
        // Handle large arrays in steps to avoid long strings on the stack
        for (let i = 0; i < len; i += 4096) {
            let part = this.rQshiftBytes(Math.min(4096, len - i), false);
            str += String.fromCharCode.apply(null, part);
        }
        return str;
    }

    rQshiftBytes(len, copy=true) {
        this._rQi += len;
        if (copy) {
            return this._rQ.slice(this._rQi - len, this._rQi);
        } else {
            return this._rQ.subarray(this._rQi - len, this._rQi);
        }
    }

    rQshiftTo(target, len) {
        // TODO: make this just use set with views when using a ArrayBuffer to store the rQ
        target.set(new Uint8Array(this._rQ.buffer, this._rQi, len));
        this._rQi += len;
    }

    rQpeekBytes(len, copy=true) {
        if (copy) {
            return this._rQ.slice(this._rQi, this._rQi + len);
        } else {
            return this._rQ.subarray(this._rQi, this._rQi + len);
        }
    }

    // Check to see if we must wait for 'num' bytes (default to FBU.bytes)
    // to be available in the receive queue. Return true if we need to
    // wait (and possibly print a debug message), otherwise false.
    rQwait(msg, num, goback) {
        if (this._rQlen - this._rQi < num) {
            if (goback) {
                if (this._rQi < goback) {
                    throw new Error("rQwait cannot backup " + goback + " bytes");
                }
                this._rQi -= goback;
            }
            return true; // true means need more data
        }
        return false;
    }

    // Send queue

    sQpush8(num) {
        this._sQensureSpace(1);
        this._sQ[this._sQlen++] = num;
    }

    sQpush16(num) {
        this._sQensureSpace(2);
        this._sQ[this._sQlen++] = (num >> 8) & 0xff;
        this._sQ[this._sQlen++] = (num >> 0) & 0xff;
    }

    sQpush32(num) {
        this._sQensureSpace(4);
        this._sQ[this._sQlen++] = (num >> 24) & 0xff;
        this._sQ[this._sQlen++] = (num >> 16) & 0xff;
        this._sQ[this._sQlen++] = (num >>  8) & 0xff;
        this._sQ[this._sQlen++] = (num >>  0) & 0xff;
    }

    sQpushString(str) {
        let bytes = str.split('').map(chr => chr.charCodeAt(0));
        this.sQpushBytes(new Uint8Array(bytes));
    }

    sQpushBytes(bytes) {
        for (let offset = 0;offset < bytes.length;) {
            this._sQensureSpace(1);

            let chunkSize = this._sQbufferSize - this._sQlen;
            if (chunkSize > bytes.length - offset) {
                chunkSize = bytes.length - offset;
            }

            this._sQ.set(bytes.subarray(offset, offset + chunkSize), this._sQlen);
            this._sQlen += chunkSize;
            offset += chunkSize;
        }
    }

    flush() {
        if (this._sQlen > 0 && this.readyState === 'open') {
            this._websocket.send(new Uint8Array(this._sQ.buffer, 0, this._sQlen));
            this._sQlen = 0;
        }
    }

    _sQensureSpace(bytes) {
        if (this._sQbufferSize - this._sQlen < bytes) {
            this.flush();
        }
    }

    // Event handlers
    off(evt) {
        this._eventHandlers[evt] = () => {};
    }

    on(evt, handler) {
        this._eventHandlers[evt] = handler;
    }

    _allocateBuffers() {
        this._rQ = new Uint8Array(this._rQbufferSize);
        this._sQ = new Uint8Array(this._sQbufferSize);
    }

    init() {
        this._allocateBuffers();
        this._rQi = 0;
        this._websocket = null;
    }

    open(uri, protocols) {
        this.attach(new WebSocket(uri, protocols));
    }

    attach(rawChannel) {
        this.init();

        // Must get object and class methods to be compatible with the tests.
        const channelProps = [...Object.keys(rawChannel), ...Object.getOwnPropertyNames(Object.getPrototypeOf(rawChannel))];
        for (let i = 0; i < rawChannelProps.length; i++) {
            const prop = rawChannelProps[i];
            if (channelProps.indexOf(prop) < 0) {
                throw new Error('Raw channel missing property: ' + prop);
            }
        }

        this._websocket = rawChannel;
        this._websocket.binaryType = "arraybuffer";
        this._websocket.onmessage = this._recvMessage.bind(this);

        this._websocket.onopen = () => {
            Log.Debug('>> WebSock.onopen');
            if (this._websocket.protocol) {
                Log.Info("Server choose sub-protocol: " + this._websocket.protocol);
            }

            this._eventHandlers.open();
            Log.Debug("<< WebSock.onopen");
        };

        this._websocket.onclose = (e) => {
            Log.Debug(">> WebSock.onclose");
            this._eventHandlers.close(e);
            Log.Debug("<< WebSock.onclose");
        };

        this._websocket.onerror = (e) => {
            Log.Debug(">> WebSock.onerror: " + e);
            this._eventHandlers.error(e);
            Log.Debug("<< WebSock.onerror: " + e);
        };
    }

    close() {
        if (this._websocket) {
            if (this.readyState === 'connecting' ||
                this.readyState === 'open') {
                Log.Info("Closing WebSocket connection");
                this._websocket.close();
            }

            this._websocket.onmessage = () => {};
        }
    }

    // private methods

    // We want to move all the unread data to the start of the queue,
    // e.g. compacting.
    // The function also expands the receive que if needed, and for
    // performance reasons we combine these two actions to avoid
    // unnecessary copying.
    _expandCompactRQ(minFit) {
        // if we're using less than 1/8th of the buffer even with the incoming bytes, compact in place
        // instead of resizing
        const requiredBufferSize =  (this._rQlen - this._rQi + minFit) * 8;
        const resizeNeeded = this._rQbufferSize < requiredBufferSize;

        if (resizeNeeded) {
            // Make sure we always *at least* double the buffer size, and have at least space for 8x
            // the current amount of data
            this._rQbufferSize = Math.max(this._rQbufferSize * 2, requiredBufferSize);
        }

        // we don't want to grow unboundedly
        if (this._rQbufferSize > MAX_RQ_GROW_SIZE) {
            this._rQbufferSize = MAX_RQ_GROW_SIZE;
            if (this._rQbufferSize - (this._rQlen - this._rQi) < minFit) {
                throw new Error("Receive queue buffer exceeded " + MAX_RQ_GROW_SIZE + " bytes, and the new message could not fit");
            }
        }

        if (resizeNeeded) {
            const oldRQbuffer = this._rQ.buffer;
            this._rQ = new Uint8Array(this._rQbufferSize);
            this._rQ.set(new Uint8Array(oldRQbuffer, this._rQi, this._rQlen - this._rQi));
        } else {
            this._rQ.copyWithin(0, this._rQi, this._rQlen);
        }

        this._rQlen = this._rQlen - this._rQi;
        this._rQi = 0;
    }

    // push arraybuffer values onto the end of the receive que
    _recvMessage(e) {
        if (this._rQlen == this._rQi) {
            // All data has now been processed, this means we
            // can reset the receive queue.
            this._rQlen = 0;
            this._rQi = 0;
        }
        const u8 = new Uint8Array(e.data);
        if (u8.length > this._rQbufferSize - this._rQlen) {
            this._expandCompactRQ(u8.length);
        }
        this._rQ.set(u8, this._rQlen);
        this._rQlen += u8.length;

        if (this._rQlen - this._rQi > 0) {
            this._eventHandlers.message();
        } else {
            Log.Debug("Ignoring empty message");
        }
    }
}
