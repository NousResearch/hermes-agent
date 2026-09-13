/*
 * noVNC: HTML5 VNC client
 * Copyright (C) 2020 The noVNC authors
 * Licensed under MPL 2.0 (see LICENSE.txt)
 *
 * See README.md for usage and integration instructions.
 *
 */

const GH_NOGESTURE = 0;
const GH_ONETAP    = 1;
const GH_TWOTAP    = 2;
const GH_THREETAP  = 4;
const GH_DRAG      = 8;
const GH_LONGPRESS = 16;
const GH_TWODRAG   = 32;
const GH_PINCH     = 64;

const GH_INITSTATE = 127;

const GH_MOVE_THRESHOLD = 50;
const GH_ANGLE_THRESHOLD = 90; // Degrees

// Timeout when waiting for gestures (ms)
const GH_MULTITOUCH_TIMEOUT = 250;

// Maximum time between press and release for a tap (ms)
const GH_TAP_TIMEOUT = 1000;

// Timeout when waiting for longpress (ms)
const GH_LONGPRESS_TIMEOUT = 1000;

// Timeout when waiting to decide between PINCH and TWODRAG (ms)
const GH_TWOTOUCH_TIMEOUT = 50;

export default class GestureHandler {
    constructor() {
        this._target = null;

        this._state = GH_INITSTATE;

        this._tracked = [];
        this._ignored = [];

        this._waitingRelease = false;
        this._releaseStart = 0.0;

        this._longpressTimeoutId = null;
        this._twoTouchTimeoutId = null;

        this._boundEventHandler = this._eventHandler.bind(this);
    }

    attach(target) {
        this.detach();

        this._target = target;
        this._target.addEventListener('touchstart',
                                      this._boundEventHandler);
        this._target.addEventListener('touchmove',
                                      this._boundEventHandler);
        this._target.addEventListener('touchend',
                                      this._boundEventHandler);
        this._target.addEventListener('touchcancel',
                                      this._boundEventHandler);
    }

    detach() {
        if (!this._target) {
            return;
        }

        this._stopLongpressTimeout();
        this._stopTwoTouchTimeout();

        this._target.removeEventListener('touchstart',
                                         this._boundEventHandler);
        this._target.removeEventListener('touchmove',
                                         this._boundEventHandler);
        this._target.removeEventListener('touchend',
                                         this._boundEventHandler);
        this._target.removeEventListener('touchcancel',
                                         this._boundEventHandler);
        this._target = null;
    }

    _eventHandler(e) {
        let fn;

        e.stopPropagation();
        e.preventDefault();

        switch (e.type) {
            case 'touchstart':
                fn = this._touchStart;
                break;
            case 'touchmove':
                fn = this._touchMove;
                break;
            case 'touchend':
            case 'touchcancel':
                fn = this._touchEnd;
                break;
        }

        for (let i = 0; i < e.changedTouches.length; i++) {
            let touch = e.changedTouches[i];
            fn.call(this, touch.identifier, touch.clientX, touch.clientY);
        }
    }

    _touchStart(id, x, y) {
        // Ignore any new touches if there is already an active gesture,
        // or we're in a cleanup state
        if (this._hasDetectedGesture() || (this._state === GH_NOGESTURE)) {
            this._ignored.push(id);
            return;
        }

        // Did it take too long between touches that we should no longer
        // consider this a single gesture?
        if ((this._tracked.length > 0) &&
            ((Date.now() - this._tracked[0].started) > GH_MULTITOUCH_TIMEOUT)) {
            this._state = GH_NOGESTURE;
            this._ignored.push(id);
            return;
        }

        // If we're waiting for fingers to release then we should no longer
        // recognize new touches
        if (this._waitingRelease) {
            this._state = GH_NOGESTURE;
            this._ignored.push(id);
            return;
        }

        this._tracked.push({
            id: id,
            started: Date.now(),
            active: true,
            firstX: x,
            firstY: y,
            lastX: x,
            lastY: y,
            angle: 0
        });

        switch (this._tracked.length) {
            case 1:
                this._startLongpressTimeout();
                break;

            case 2:
                this._state &= ~(GH_ONETAP | GH_DRAG | GH_LONGPRESS);
                this._stopLongpressTimeout();
                break;

            case 3:
                this._state &= ~(GH_TWOTAP | GH_TWODRAG | GH_PINCH);
                break;

            default:
                this._state = GH_NOGESTURE;
        }
    }

    _touchMove(id, x, y) {
        let touch = this._tracked.find(t => t.id === id);

        // If this is an update for a touch we're not tracking, ignore it
        if (touch === undefined) {
            return;
        }

        // Update the touches last position with the event coordinates
        touch.lastX = x;
        touch.lastY = y;

        let deltaX = x - touch.firstX;
        let deltaY = y - touch.firstY;

        // Update angle when the touch has moved
        if ((touch.firstX !== touch.lastX) ||
            (touch.firstY !== touch.lastY)) {
            touch.angle = Math.atan2(deltaY, deltaX) * 180 / Math.PI;
        }

        if (!this._hasDetectedGesture()) {
            // Ignore moves smaller than the minimum threshold
            if (Math.hypot(deltaX, deltaY) < GH_MOVE_THRESHOLD) {
                return;
            }

            // Can't be a tap or long press as we've seen movement
            this._state &= ~(GH_ONETAP | GH_TWOTAP | GH_THREETAP | GH_LONGPRESS);
            this._stopLongpressTimeout();

            if (this._tracked.length !== 1) {
                this._state &= ~(GH_DRAG);
            }
            if (this._tracked.length !== 2) {
                this._state &= ~(GH_TWODRAG | GH_PINCH);
            }

            // We need to figure out which of our different two touch gestures
            // this might be
            if (this._tracked.length === 2) {

                // The other touch is the one where the id doesn't match
                let prevTouch = this._tracked.find(t => t.id !== id);

                // How far the previous touch point has moved since start
                let prevDeltaMove = Math.hypot(prevTouch.firstX - prevTouch.lastX,
                                               prevTouch.firstY - prevTouch.lastY);

                // We know that the current touch moved far enough,
                // but unless both touches moved further than their
                // threshold we don't want to disqualify any gestures
                if (prevDeltaMove > GH_MOVE_THRESHOLD) {

                    // The angle difference between the direction of the touch points
                    let deltaAngle = Math.abs(touch.angle - prevTouch.angle);
                    deltaAngle = Math.abs(((deltaAngle + 180) % 360) - 180);

                    // PINCH or TWODRAG can be eliminated depending on the angle
                    if (deltaAngle > GH_ANGLE_THRESHOLD) {
                        this._state &= ~GH_TWODRAG;
                    } else {
                        this._state &= ~GH_PINCH;
                    }

                    if (this._isTwoTouchTimeoutRunning()) {
                        this._stopTwoTouchTimeout();
                    }
                } else if (!this._isTwoTouchTimeoutRunning()) {
                    // We can't determine the gesture right now, let's
                    // wait and see if more events are on their way
                    this._startTwoTouchTimeout();
                }
            }

            if (!this._hasDetectedGesture()) {
                return;
            }

            this._pushEvent('gesturestart');
        }

        this._pushEvent('gesturemove');
    }

    _touchEnd(id, x, y) {
        // Check if this is an ignored touch
        if (this._ignored.indexOf(id) !== -1) {
            // Remove this touch from ignored
            this._ignored.splice(this._ignored.indexOf(id), 1);

            // And reset the state if there are no more touches
            if ((this._ignored.length === 0) &&
                (this._tracked.length === 0)) {
                this._state = GH_INITSTATE;
                this._waitingRelease = false;
            }
            return;
        }

        // We got a touchend before the timer triggered,
        // this cannot result in a gesture anymore.
        if (!this._hasDetectedGesture() &&
            this._isTwoTouchTimeoutRunning()) {
            this._stopTwoTouchTimeout();
            this._state = GH_NOGESTURE;
        }

        // Some gestures don't trigger until a touch is released
        if (!this._hasDetectedGesture()) {
            // Can't be a gesture that relies on movement
            this._state &= ~(GH_DRAG | GH_TWODRAG | GH_PINCH);
            // Or something that relies on more time
            this._state &= ~GH_LONGPRESS;
            this._stopLongpressTimeout();

            if (!this._waitingRelease) {
                this._releaseStart = Date.now();
                this._waitingRelease = true;

                // Can't be a tap that requires more touches than we current have
                switch (this._tracked.length) {
                    case 1:
                        this._state &= ~(GH_TWOTAP | GH_THREETAP);
                        break;

                    case 2:
                        this._state &= ~(GH_ONETAP | GH_THREETAP);
                        break;
                }
            }
        }

        // Waiting for all touches to release? (i.e. some tap)
        if (this._waitingRelease) {
            // Were all touches released at roughly the same time?
            if ((Date.now() - this._releaseStart) > GH_MULTITOUCH_TIMEOUT) {
                this._state = GH_NOGESTURE;
            }

            // Did too long time pass between press and release?
            if (this._tracked.some(t => (Date.now() - t.started) > GH_TAP_TIMEOUT)) {
                this._state = GH_NOGESTURE;
            }

            let touch = this._tracked.find(t => t.id === id);
            touch.active = false;

            // Are we still waiting for more releases?
            if (this._hasDetectedGesture()) {
                this._pushEvent('gesturestart');
            } else {
                // Have we reached a dead end?
                if (this._state !== GH_NOGESTURE) {
                    return;
                }
            }
        }

        if (this._hasDetectedGesture()) {
            this._pushEvent('gestureend');
        }

        // Ignore any remaining touches until they are ended
        for (let i = 0; i < this._tracked.length; i++) {
            if (this._tracked[i].active) {
                this._ignored.push(this._tracked[i].id);
            }
        }
        this._tracked = [];

        this._state = GH_NOGESTURE;

        // Remove this touch from ignored if it's in there
        if (this._ignored.indexOf(id) !== -1) {
            this._ignored.splice(this._ignored.indexOf(id), 1);
        }

        // We reset the state if ignored is empty
        if ((this._ignored.length === 0)) {
            this._state = GH_INITSTATE;
            this._waitingRelease = false;
        }
    }

    _hasDetectedGesture() {
        if (this._state === GH_NOGESTURE) {
            return false;
        }
        // Check to see if the bitmask value is a power of 2
        // (i.e. only one bit set). If it is, we have a state.
        if (this._state & (this._state - 1)) {
            return false;
        }

        // For taps we also need to have all touches released
        // before we've fully detected the gesture
        if (this._state & (GH_ONETAP | GH_TWOTAP | GH_THREETAP)) {
            if (this._tracked.some(t => t.active)) {
                return false;
            }
        }

        return true;
    }

    _startLongpressTimeout() {
        this._stopLongpressTimeout();
        this._longpressTimeoutId = setTimeout(() => this._longpressTimeout(),
                                              GH_LONGPRESS_TIMEOUT);
    }

    _stopLongpressTimeout() {
        clearTimeout(this._longpressTimeoutId);
        this._longpressTimeoutId = null;
    }

    _longpressTimeout() {
        if (this._hasDetectedGesture()) {
            throw new Error("A longpress gesture failed, conflict with a different gesture");
        }

        this._state = GH_LONGPRESS;
        this._pushEvent('gesturestart');
    }

    _startTwoTouchTimeout() {
        this._stopTwoTouchTimeout();
        this._twoTouchTimeoutId = setTimeout(() => this._twoTouchTimeout(),
                                             GH_TWOTOUCH_TIMEOUT);
    }

    _stopTwoTouchTimeout() {
        clearTimeout(this._twoTouchTimeoutId);
        this._twoTouchTimeoutId = null;
    }

    _isTwoTouchTimeoutRunning() {
        return this._twoTouchTimeoutId !== null;
    }

    _twoTouchTimeout() {
        if (this._tracked.length === 0) {
            throw new Error("A pinch or two drag gesture failed, no tracked touches");
        }

        // How far each touch point has moved since start
        let avgM = this._getAverageMovement();
        let avgMoveH = Math.abs(avgM.x);
        let avgMoveV = Math.abs(avgM.y);

        // The difference in the distance between where
        // the touch points started and where they are now
        let avgD = this._getAverageDistance();
        let deltaTouchDistance = Math.abs(Math.hypot(avgD.first.x, avgD.first.y) -
                                          Math.hypot(avgD.last.x, avgD.last.y));

        if ((avgMoveV < deltaTouchDistance) &&
            (avgMoveH < deltaTouchDistance)) {
            this._state = GH_PINCH;
        } else {
            this._state = GH_TWODRAG;
        }

        this._pushEvent('gesturestart');
        this._pushEvent('gesturemove');
    }

    _pushEvent(type) {
        let detail = { type: this._stateToGesture(this._state) };

        // For most gesture events the current (average) position is the
        // most useful
        let avg = this._getPosition();
        let pos = avg.last;

        // However we have a slight distance to detect gestures, so for the
        // first gesture event we want to use the first positions we saw
        if (type === 'gesturestart') {
            pos = avg.first;
        }

        // For these gestures, we always want the event coordinates
        // to be where the gesture began, not the current touch location.
        switch (this._state) {
            case GH_TWODRAG:
            case GH_PINCH:
                pos = avg.first;
                break;
        }

        detail['clientX'] = pos.x;
        detail['clientY'] = pos.y;

        // FIXME: other coordinates?

        // Some gestures also have a magnitude
        if (this._state === GH_PINCH) {
            let distance = this._getAverageDistance();
            if (type === 'gesturestart') {
                detail['magnitudeX'] = distance.first.x;
                detail['magnitudeY'] = distance.first.y;
            } else {
                detail['magnitudeX'] = distance.last.x;
                detail['magnitudeY'] = distance.last.y;
            }
        } else if (this._state === GH_TWODRAG) {
            if (type === 'gesturestart') {
                detail['magnitudeX'] = 0.0;
                detail['magnitudeY'] = 0.0;
            } else {
                let movement = this._getAverageMovement();
                detail['magnitudeX'] = movement.x;
                detail['magnitudeY'] = movement.y;
            }
        }

        let gev = new CustomEvent(type, { detail: detail });
        this._target.dispatchEvent(gev);
    }

    _stateToGesture(state) {
        switch (state) {
            case GH_ONETAP:
                return 'onetap';
            case GH_TWOTAP:
                return 'twotap';
            case GH_THREETAP:
                return 'threetap';
            case GH_DRAG:
                return 'drag';
            case GH_LONGPRESS:
                return 'longpress';
            case GH_TWODRAG:
                return 'twodrag';
            case GH_PINCH:
                return 'pinch';
        }

        throw new Error("Unknown gesture state: " + state);
    }

    _getPosition() {
        if (this._tracked.length === 0) {
            throw new Error("Failed to get gesture position, no tracked touches");
        }

        let size = this._tracked.length;
        let fx = 0, fy = 0, lx = 0, ly = 0;

        for (let i = 0; i < this._tracked.length; i++) {
            fx += this._tracked[i].firstX;
            fy += this._tracked[i].firstY;
            lx += this._tracked[i].lastX;
            ly += this._tracked[i].lastY;
        }

        return { first: { x: fx / size,
                          y: fy / size },
                 last: { x: lx / size,
                         y: ly / size } };
    }

    _getAverageMovement() {
        if (this._tracked.length === 0) {
            throw new Error("Failed to get gesture movement, no tracked touches");
        }

        let totalH, totalV;
        totalH = totalV = 0;
        let size = this._tracked.length;

        for (let i = 0; i < this._tracked.length; i++) {
            totalH += this._tracked[i].lastX - this._tracked[i].firstX;
            totalV += this._tracked[i].lastY - this._tracked[i].firstY;
        }

        return { x: totalH / size,
                 y: totalV / size };
    }

    _getAverageDistance() {
        if (this._tracked.length === 0) {
            throw new Error("Failed to get gesture distance, no tracked touches");
        }

        // Distance between the first and last tracked touches

        let first = this._tracked[0];
        let last = this._tracked[this._tracked.length - 1];

        let fdx = Math.abs(last.firstX - first.firstX);
        let fdy = Math.abs(last.firstY - first.firstY);

        let ldx = Math.abs(last.lastX - first.lastX);
        let ldy = Math.abs(last.lastY - first.lastY);

        return { first: { x: fdx, y: fdy },
                 last: { x: ldx, y: ldy } };
    }
}
