/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2024 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 *
 */

import * as Log from '../util/logging.js';

export class H264Parser {
    constructor(data) {
        this._data = data;
        this._index = 0;
        this.profileIdc = null;
        this.constraintSet = null;
        this.levelIdc = null;
    }

    _getStartSequenceLen(index) {
        let data = this._data;
        if (data[index + 0] == 0 && data[index + 1] == 0 && data[index + 2] == 0 && data[index + 3] == 1) {
            return 4;
        }
        if (data[index + 0] == 0 && data[index + 1] == 0 && data[index + 2] == 1) {
            return 3;
        }
        return 0;
    }

    _indexOfNextNalUnit(index) {
        let data = this._data;
        for (let i = index; i < data.length; ++i) {
            if (this._getStartSequenceLen(i) != 0) {
                return i;
            }
        }
        return -1;
    }

    _parseSps(index) {
        this.profileIdc = this._data[index];
        this.constraintSet = this._data[index + 1];
        this.levelIdc = this._data[index + 2];
    }

    _parseNalUnit(index) {
        const firstByte = this._data[index];
        if (firstByte & 0x80) {
            throw new Error('H264 parsing sanity check failed, forbidden zero bit is set');
        }
        const unitType = firstByte & 0x1f;

        switch (unitType) {
            case 1: // coded slice, non-idr
                return { slice: true };
            case 5: // coded slice, idr
                return { slice: true, key: true };
            case 6: // sei
                return {};
            case 7: // sps
                this._parseSps(index + 1);
                return {};
            case 8: // pps
                return {};
            default:
                Log.Warn("Unhandled unit type: ", unitType);
                break;
        }
        return {};
    }

    parse() {
        const startIndex = this._index;
        let isKey = false;

        while (this._index < this._data.length) {
            const startSequenceLen = this._getStartSequenceLen(this._index);
            if (startSequenceLen == 0) {
                throw new Error('Invalid start sequence in bit stream');
            }

            const { slice, key } = this._parseNalUnit(this._index + startSequenceLen);

            let nextIndex = this._indexOfNextNalUnit(this._index + startSequenceLen);
            if (nextIndex == -1) {
                this._index = this._data.length;
            } else {
                this._index = nextIndex;
            }

            if (key) {
                isKey = true;
            }
            if (slice) {
                break;
            }
        }

        if (startIndex === this._index) {
            return null;
        }

        return {
            frame: this._data.subarray(startIndex, this._index),
            key: isKey,
        };
    }
}

export class H264Context {
    constructor(width, height) {
        this.lastUsed = 0;
        this._width = width;
        this._height = height;
        this._profileIdc = null;
        this._constraintSet = null;
        this._levelIdc = null;
        this._decoder = null;
        this._pendingFrames = [];
    }

    _handleFrame(frame) {
        let pending = this._pendingFrames.shift();
        if (pending === undefined) {
            throw new Error("Pending frame queue empty when receiving frame from decoder");
        }

        if (pending.timestamp != frame.timestamp) {
            throw new Error("Video frame timestamp mismatch. Expected " +
                frame.timestamp + " but but got " + pending.timestamp);
        }

        pending.frame = frame;
        pending.ready = true;
        pending.resolve();

        if (!pending.keep) {
            frame.close();
        }
    }

    _handleError(e) {
        throw new Error("Failed to decode frame: " + e.message);
    }

    _configureDecoder(profileIdc, constraintSet, levelIdc) {
        if (this._decoder === null || this._decoder.state === 'closed') {
            this._decoder = new VideoDecoder({
                output: frame => this._handleFrame(frame),
                error: e => this._handleError(e),
            });
        }
        const codec = 'avc1.' +
            profileIdc.toString(16).padStart(2, '0') +
            constraintSet.toString(16).padStart(2, '0') +
            levelIdc.toString(16).padStart(2, '0');
        this._decoder.configure({
            codec: codec,
            codedWidth: this._width,
            codedHeight: this._height,
            optimizeForLatency: true,
        });
    }

    _preparePendingFrame(timestamp) {
        let pending = {
            timestamp: timestamp,
            promise: null,
            resolve: null,
            frame: null,
            ready: false,
            keep: false,
        };
        pending.promise = new Promise((resolve) => {
            pending.resolve = resolve;
        });
        this._pendingFrames.push(pending);

        return pending;
    }

    decode(payload) {
        let parser = new H264Parser(payload);
        let result = null;

        // Ideally, this timestamp should come from the server, but we'll just
        // approximate it instead.
        let timestamp = Math.round(window.performance.now() * 1e3);

        while (true) {
            let encodedFrame = parser.parse();
            if (encodedFrame === null) {
                break;
            }

            if (parser.profileIdc !== null) {
                self._profileIdc = parser.profileIdc;
                self._constraintSet = parser.constraintSet;
                self._levelIdc = parser.levelIdc;
            }

            if (this._decoder === null || this._decoder.state !== 'configured') {
                if (!encodedFrame.key) {
                    Log.Warn("Missing key frame. Can't decode until one arrives");
                    continue;
                }
                if (self._profileIdc === null) {
                    Log.Warn('Cannot config decoder. Have not received SPS and PPS yet.');
                    continue;
                }
                this._configureDecoder(self._profileIdc, self._constraintSet,
                                       self._levelIdc);
            }

            result = this._preparePendingFrame(timestamp);

            const chunk = new EncodedVideoChunk({
                timestamp: timestamp,
                type: encodedFrame.key ? 'key' : 'delta',
                data: encodedFrame.frame,
            });

            try {
                this._decoder.decode(chunk);
            } catch (e) {
                Log.Warn("Failed to decode:", e);
            }
        }

        // We only keep last frame of each payload
        if (result !== null) {
            result.keep = true;
        }

        return result;
    }
}

export default class H264Decoder {
    constructor() {
        this._tick = 0;
        this._contexts = {};
    }

    _contextId(x, y, width, height) {
        return [x, y, width, height].join(',');
    }

    _findOldestContextId() {
        let oldestTick = Number.MAX_VALUE;
        let oldestKey = undefined;
        for (const [key, value] of Object.entries(this._contexts)) {
            if (value.lastUsed < oldestTick) {
                oldestTick = value.lastUsed;
                oldestKey = key;
            }
        }
        return oldestKey;
    }

    _createContext(x, y, width, height) {
        const maxContexts = 64;
        if (Object.keys(this._contexts).length >= maxContexts) {
            let oldestContextId = this._findOldestContextId();
            delete this._contexts[oldestContextId];
        }
        let context = new H264Context(width, height);
        this._contexts[this._contextId(x, y, width, height)] = context;
        return context;
    }

    _getContext(x, y, width, height) {
        let context = this._contexts[this._contextId(x, y, width, height)];
        return context !== undefined ? context : this._createContext(x, y, width, height);
    }

    _resetContext(x, y, width, height) {
        delete this._contexts[this._contextId(x, y, width, height)];
    }

    _resetAllContexts() {
        this._contexts = {};
    }

    decodeRect(x, y, width, height, sock, display, depth) {
        const resetContextFlag = 1;
        const resetAllContextsFlag = 2;

        if (sock.rQwait("h264 header", 8)) {
            return false;
        }

        const length = sock.rQshift32();
        const flags = sock.rQshift32();

        if (sock.rQwait("h264 payload", length, 8)) {
            return false;
        }

        if (flags & resetAllContextsFlag) {
            this._resetAllContexts();
        } else if (flags & resetContextFlag) {
            this._resetContext(x, y, width, height);
        }

        let context = this._getContext(x, y, width, height);
        context.lastUsed = this._tick++;

        if (length !== 0) {
            let payload = sock.rQshiftBytes(length, false);
            let frame = context.decode(payload);
            if (frame !== null) {
                display.videoFrame(x, y, width, height, frame);
            }
        }

        return true;
    }
}
