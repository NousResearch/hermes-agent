import KeyTable from "./keysym.js";
import keysyms from "./keysymdef.js";
import vkeys from "./vkeys.js";
import fixedkeys from "./fixedkeys.js";
import DOMKeyTable from "./domkeytable.js";
import * as browser from "../util/browser.js";

// Get 'KeyboardEvent.code', handling legacy browsers
export function getKeycode(evt) {
    // Are we getting proper key identifiers?
    // (unfortunately Firefox and Chrome are crappy here and gives
    // us an empty string on some platforms, rather than leaving it
    // undefined)
    if (evt.code) {
        // Mozilla isn't fully in sync with the spec yet
        switch (evt.code) {
            case 'OSLeft': return 'MetaLeft';
            case 'OSRight': return 'MetaRight';
        }

        return evt.code;
    }

    // The de-facto standard is to use Windows Virtual-Key codes
    // in the 'keyCode' field for non-printable characters
    if (evt.keyCode in vkeys) {
        let code = vkeys[evt.keyCode];

        // macOS has messed up this code for some reason
        if (browser.isMac() && (code === 'ContextMenu')) {
            code = 'MetaRight';
        }

        // The keyCode doesn't distinguish between left and right
        // for the standard modifiers
        if (evt.location === 2) {
            switch (code) {
                case 'ShiftLeft': return 'ShiftRight';
                case 'ControlLeft': return 'ControlRight';
                case 'AltLeft': return 'AltRight';
            }
        }

        // Nor a bunch of the numpad keys
        if (evt.location === 3) {
            switch (code) {
                case 'Delete': return 'NumpadDecimal';
                case 'Insert': return 'Numpad0';
                case 'End': return 'Numpad1';
                case 'ArrowDown': return 'Numpad2';
                case 'PageDown': return 'Numpad3';
                case 'ArrowLeft': return 'Numpad4';
                case 'ArrowRight': return 'Numpad6';
                case 'Home': return 'Numpad7';
                case 'ArrowUp': return 'Numpad8';
                case 'PageUp': return 'Numpad9';
                case 'Enter': return 'NumpadEnter';
            }
        }

        return code;
    }

    return 'Unidentified';
}

// Get 'KeyboardEvent.key', handling legacy browsers
export function getKey(evt) {
    // Are we getting a proper key value?
    if ((evt.key !== undefined) && (evt.key !== 'Unidentified')) {
        // Mozilla isn't fully in sync with the spec yet
        switch (evt.key) {
            case 'OS': return 'Meta';
            case 'LaunchMyComputer': return 'LaunchApplication1';
            case 'LaunchCalculator': return 'LaunchApplication2';
        }

        // iOS leaks some OS names
        switch (evt.key) {
            case 'UIKeyInputUpArrow': return 'ArrowUp';
            case 'UIKeyInputDownArrow': return 'ArrowDown';
            case 'UIKeyInputLeftArrow': return 'ArrowLeft';
            case 'UIKeyInputRightArrow': return 'ArrowRight';
            case 'UIKeyInputEscape': return 'Escape';
        }

        // Broken behaviour in Chrome
        if ((evt.key === '\x00') && (evt.code === 'NumpadDecimal')) {
            return 'Delete';
        }

        return evt.key;
    }

    // Try to deduce it based on the physical key
    const code = getKeycode(evt);
    if (code in fixedkeys) {
        return fixedkeys[code];
    }

    // If that failed, then see if we have a printable character
    if (evt.charCode) {
        return String.fromCharCode(evt.charCode);
    }

    // At this point we have nothing left to go on
    return 'Unidentified';
}

// Get the most reliable keysym value we can get from a key event
export function getKeysym(evt) {
    const key = getKey(evt);

    if (key === 'Unidentified') {
        return null;
    }

    // First look up special keys
    if (key in DOMKeyTable) {
        let location = evt.location;

        // Safari screws up location for the right cmd key
        if ((key === 'Meta') && (location === 0)) {
            location = 2;
        }

        // And for Clear
        if ((key === 'Clear') && (location === 3)) {
            let code = getKeycode(evt);
            if (code === 'NumLock') {
                location = 0;
            }
        }

        if ((location === undefined) || (location > 3)) {
            location = 0;
        }

        // The original Meta key now gets confused with the Windows key
        // https://bugs.chromium.org/p/chromium/issues/detail?id=1020141
        // https://bugzilla.mozilla.org/show_bug.cgi?id=1232918
        if (key === 'Meta') {
            let code = getKeycode(evt);
            if (code === 'AltLeft') {
                return KeyTable.XK_Meta_L;
            } else if (code === 'AltRight') {
                return KeyTable.XK_Meta_R;
            }
        }

        // macOS has Clear instead of NumLock, but the remote system is
        // probably not macOS, so lying here is probably best...
        if (key === 'Clear') {
            let code = getKeycode(evt);
            if (code === 'NumLock') {
                return KeyTable.XK_Num_Lock;
            }
        }

        // Windows sends alternating symbols for some keys when using a
        // Japanese layout. We have no way of synchronising with the IM
        // running on the remote system, so we send some combined keysym
        // instead and hope for the best.
        if (browser.isWindows()) {
            switch (key) {
                case 'Zenkaku':
                case 'Hankaku':
                    return KeyTable.XK_Zenkaku_Hankaku;
                case 'Romaji':
                case 'KanaMode':
                    return KeyTable.XK_Romaji;
            }
        }

        return DOMKeyTable[key][location];
    }

    // Now we need to look at the Unicode symbol instead

    // Special key? (FIXME: Should have been caught earlier)
    if (key.length !== 1) {
        return null;
    }

    const codepoint = key.charCodeAt();
    if (codepoint) {
        return keysyms.lookup(codepoint);
    }

    return null;
}
