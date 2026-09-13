/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2020 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

export function toUnsigned32bit(toConvert) {
    return toConvert >>> 0;
}

export function toSigned32bit(toConvert) {
    return toConvert | 0;
}
