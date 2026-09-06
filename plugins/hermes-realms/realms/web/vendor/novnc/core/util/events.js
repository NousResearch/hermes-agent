/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2018 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 */

/*
 * Cross-browser event and position routines
 */

export function getPointerEvent(e) {
    return e.changedTouches ? e.changedTouches[0] : e.touches ? e.touches[0] : e;
}

export function stopEvent(e) {
    e.stopPropagation();
    e.preventDefault();
}

// Emulate Element.setCapture() when not supported
let _captureRecursion = false;
let _elementForUnflushedEvents = null;
document.captureElement = null;
function _captureProxy(e) {
    // Recursion protection as we'll see our own event
    if (_captureRecursion) return;

    // Clone the event as we cannot dispatch an already dispatched event
    const newEv = new e.constructor(e.type, e);

    _captureRecursion = true;
    if (document.captureElement) {
        document.captureElement.dispatchEvent(newEv);
    } else {
        _elementForUnflushedEvents.dispatchEvent(newEv);
    }
    _captureRecursion = false;

    // Avoid double events
    e.stopPropagation();

    // Respect the wishes of the redirected event handlers
    if (newEv.defaultPrevented) {
        e.preventDefault();
    }

    // Implicitly release the capture on button release
    if (e.type === "mouseup") {
        releaseCapture();
    }
}

// Follow cursor style of target element
function _capturedElemChanged() {
    const proxyElem = document.getElementById("noVNC_mouse_capture_elem");
    proxyElem.style.cursor = window.getComputedStyle(document.captureElement).cursor;
}

const _captureObserver = new MutationObserver(_capturedElemChanged);

export function setCapture(target) {
    if (target.setCapture) {

        target.setCapture();
        document.captureElement = target;
    } else {
        // Release any existing capture in case this method is
        // called multiple times without coordination
        releaseCapture();

        let proxyElem = document.getElementById("noVNC_mouse_capture_elem");

        if (proxyElem === null) {
            proxyElem = document.createElement("div");
            proxyElem.id = "noVNC_mouse_capture_elem";
            proxyElem.style.position = "fixed";
            proxyElem.style.top = "0px";
            proxyElem.style.left = "0px";
            proxyElem.style.width = "100%";
            proxyElem.style.height = "100%";
            proxyElem.style.zIndex = 10000;
            proxyElem.style.display = "none";
            document.body.appendChild(proxyElem);

            // This is to make sure callers don't get confused by having
            // our blocking element as the target
            proxyElem.addEventListener('contextmenu', _captureProxy);

            proxyElem.addEventListener('mousemove', _captureProxy);
            proxyElem.addEventListener('mouseup', _captureProxy);
        }

        document.captureElement = target;

        // Track cursor and get initial cursor
        _captureObserver.observe(target, {attributes: true});
        _capturedElemChanged();

        proxyElem.style.display = "";

        // We listen to events on window in order to keep tracking if it
        // happens to leave the viewport
        window.addEventListener('mousemove', _captureProxy);
        window.addEventListener('mouseup', _captureProxy);
    }
}

export function releaseCapture() {
    if (document.releaseCapture) {

        document.releaseCapture();
        document.captureElement = null;

    } else {
        if (!document.captureElement) {
            return;
        }

        // There might be events already queued. The event proxy needs
        // access to the captured element for these queued events.
        // E.g. contextmenu (right-click) in Microsoft Edge
        //
        // Before removing the capturedElem pointer we save it to a
        // temporary variable that the unflushed events can use.
        _elementForUnflushedEvents = document.captureElement;
        document.captureElement = null;

        _captureObserver.disconnect();

        const proxyElem = document.getElementById("noVNC_mouse_capture_elem");
        proxyElem.style.display = "none";

        window.removeEventListener('mousemove', _captureProxy);
        window.removeEventListener('mouseup', _captureProxy);
    }
}
