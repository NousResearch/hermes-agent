/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2019 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

// Decode from UTF-8
export function decodeUTF8(utf8string, allowLatin1=false) {
    try {
        return decodeURIComponent(escape(utf8string));
    } catch (e) {
        if (e instanceof URIError) {
            if (allowLatin1) {
                // If we allow Latin1 we can ignore any decoding fails
                // and in these cases return the original string
                return utf8string;
            }
        }
        throw e;
    }
}

// Encode to UTF-8
export function encodeUTF8(DOMString) {
    return unescape(encodeURIComponent(DOMString));
}
