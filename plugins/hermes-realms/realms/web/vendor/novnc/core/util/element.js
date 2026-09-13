/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2020 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

/*
 * HTML element utility functions
 */

export function clientToElement(x, y, elem) {
    const bounds = elem.getBoundingClientRect();
    let pos = { x: 0, y: 0 };
    // Clip to target bounds
    if (x < bounds.left) {
        pos.x = 0;
    } else if (x >= bounds.right) {
        pos.x = bounds.width - 1;
    } else {
        pos.x = x - bounds.left;
    }
    if (y < bounds.top) {
        pos.y = 0;
    } else if (y >= bounds.bottom) {
        pos.y = bounds.height - 1;
    } else {
        pos.y = y - bounds.top;
    }
    return pos;
}
