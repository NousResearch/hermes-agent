/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2018 The noVNC authors
 * Licensed under MPL 2.0 or any later version (see LICENSE.txt)
 */

/*
 * Fallback mapping between HTML key codes (physical keys) and
 * HTML key values. This only works for keys that don't vary
 * between layouts. We also omit those who manage fine by mapping the
 * Unicode representation.
 *
 * See https://www.w3.org/TR/uievents-code/ for possible codes.
 * See https://www.w3.org/TR/uievents-key/ for possible values.
 */

/* eslint-disable key-spacing */

export default {

// 3.1.1.1. Writing System Keys

    'Backspace':        'Backspace',

// 3.1.1.2. Functional Keys

    'AltLeft':          'Alt',
    'AltRight':         'Alt', // This could also be 'AltGraph'
    'CapsLock':         'CapsLock',
    'ContextMenu':      'ContextMenu',
    'ControlLeft':      'Control',
    'ControlRight':     'Control',
    'Enter':            'Enter',
    'MetaLeft':         'Meta',
    'MetaRight':        'Meta',
    'ShiftLeft':        'Shift',
    'ShiftRight':       'Shift',
    'Tab':              'Tab',
    // FIXME: Japanese/Korean keys

// 3.1.2. Control Pad Section

    'Delete':           'Delete',
    'End':              'End',
    'Help':             'Help',
    'Home':             'Home',
    'Insert':           'Insert',
    'PageDown':         'PageDown',
    'PageUp':           'PageUp',

// 3.1.3. Arrow Pad Section

    'ArrowDown':        'ArrowDown',
    'ArrowLeft':        'ArrowLeft',
    'ArrowRight':       'ArrowRight',
    'ArrowUp':          'ArrowUp',

// 3.1.4. Numpad Section

    'NumLock':          'NumLock',
    'NumpadBackspace':  'Backspace',
    'NumpadClear':      'Clear',

// 3.1.5. Function Section

    'Escape':           'Escape',
    'F1':               'F1',
    'F2':               'F2',
    'F3':               'F3',
    'F4':               'F4',
    'F5':               'F5',
    'F6':               'F6',
    'F7':               'F7',
    'F8':               'F8',
    'F9':               'F9',
    'F10':              'F10',
    'F11':              'F11',
    'F12':              'F12',
    'F13':              'F13',
    'F14':              'F14',
    'F15':              'F15',
    'F16':              'F16',
    'F17':              'F17',
    'F18':              'F18',
    'F19':              'F19',
    'F20':              'F20',
    'F21':              'F21',
    'F22':              'F22',
    'F23':              'F23',
    'F24':              'F24',
    'F25':              'F25',
    'F26':              'F26',
    'F27':              'F27',
    'F28':              'F28',
    'F29':              'F29',
    'F30':              'F30',
    'F31':              'F31',
    'F32':              'F32',
    'F33':              'F33',
    'F34':              'F34',
    'F35':              'F35',
    'PrintScreen':      'PrintScreen',
    'ScrollLock':       'ScrollLock',
    'Pause':            'Pause',

// 3.1.6. Media Keys

    'BrowserBack':      'BrowserBack',
    'BrowserFavorites': 'BrowserFavorites',
    'BrowserForward':   'BrowserForward',
    'BrowserHome':      'BrowserHome',
    'BrowserRefresh':   'BrowserRefresh',
    'BrowserSearch':    'BrowserSearch',
    'BrowserStop':      'BrowserStop',
    'Eject':            'Eject',
    'LaunchApp1':       'LaunchMyComputer',
    'LaunchApp2':       'LaunchCalendar',
    'LaunchMail':       'LaunchMail',
    'MediaPlayPause':   'MediaPlay',
    'MediaStop':        'MediaStop',
    'MediaTrackNext':   'MediaTrackNext',
    'MediaTrackPrevious': 'MediaTrackPrevious',
    'Power':            'Power',
    'Sleep':            'Sleep',
    'AudioVolumeDown':  'AudioVolumeDown',
    'AudioVolumeMute':  'AudioVolumeMute',
    'AudioVolumeUp':    'AudioVolumeUp',
    'WakeUp':           'WakeUp',
};
