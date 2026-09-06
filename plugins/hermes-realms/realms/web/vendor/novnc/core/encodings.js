/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2019 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

export const encodings = {
    encodingRaw: 0,
    encodingCopyRect: 1,
    encodingRRE: 2,
    encodingHextile: 5,
    encodingZlib: 6,
    encodingTight: 7,
    encodingZRLE: 16,
    encodingTightPNG: -260,
    encodingJPEG: 21,
    encodingH264: 50,

    pseudoEncodingQualityLevel9: -23,
    pseudoEncodingQualityLevel0: -32,
    pseudoEncodingDesktopSize: -223,
    pseudoEncodingLastRect: -224,
    pseudoEncodingCursor: -239,
    pseudoEncodingQEMUExtendedKeyEvent: -258,
    pseudoEncodingQEMULedEvent: -261,
    pseudoEncodingDesktopName: -307,
    pseudoEncodingExtendedDesktopSize: -308,
    pseudoEncodingXvp: -309,
    pseudoEncodingFence: -312,
    pseudoEncodingContinuousUpdates: -313,
    pseudoEncodingExtendedMouseButtons: -316,
    pseudoEncodingCompressLevel9: -247,
    pseudoEncodingCompressLevel0: -256,
    pseudoEncodingVMwareCursor: 0x574d5664,
    pseudoEncodingExtendedClipboard: 0xc0a1e5ce
};

export function encodingName(num) {
    switch (num) {
        case encodings.encodingRaw:      return "Raw";
        case encodings.encodingCopyRect: return "CopyRect";
        case encodings.encodingRRE:      return "RRE";
        case encodings.encodingHextile:  return "Hextile";
        case encodings.encodingZlib:     return "Zlib";
        case encodings.encodingTight:    return "Tight";
        case encodings.encodingZRLE:     return "ZRLE";
        case encodings.encodingTightPNG: return "TightPNG";
        case encodings.encodingJPEG:     return "JPEG";
        case encodings.encodingH264:     return "H.264";
        default:                         return "[unknown encoding " + num + "]";
    }
}
