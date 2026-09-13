"""Incremental RFB 3.8 client parser: view-only is enforced off-browser.

The private wayvnc listener uses RFB None security; authorization is owned by
our WebSocket capability gate. Unknown extensions fail closed, not pass through.
"""


class ClientFilter:
    MAX_PAYLOAD = 1_048_576
    INPUT = frozenset({4, 5, 6, 251, 255})
    FIXED = {0: 20, 3: 10, 4: 8, 5: 6, 150: 10, 255: 12}

    def __init__(self, *, control=False):
        self.control = control
        self._buffer = bytearray()
        self._stage = 0

    def feed(self, data):
        if len(self._buffer) + len(data) > self.MAX_PAYLOAD + 24:
            raise ValueError("RFB message too large")
        self._buffer.extend(data)
        output = bytearray()
        while self._buffer:
            if self._stage < 3:
                length = (12, 1, 1)[self._stage]
                if len(self._buffer) < length:
                    break
                frame = bytes(self._buffer[:length])
                valid = (
                    frame == b"RFB 003.008\n"
                    if self._stage == 0
                    else frame == b"\x01"
                    if self._stage == 1
                    else frame in (b"\x00", b"\x01")
                )
                if not valid:
                    raise ValueError("Unsupported RFB handshake")
                self._stage += 1
                output.extend(frame)
            else:
                kind = self._buffer[0]
                length = self._length(kind)
                if length is None or len(self._buffer) < length:
                    break
                if self.control or kind not in self.INPUT:
                    output.extend(self._buffer[:length])
            del self._buffer[:length]
        return bytes(output)

    def _length(self, kind):
        if kind in self.FIXED:
            return self.FIXED[kind]
        layouts = {
            1: (6, 4, 2, 6),
            2: (4, 2, 2, 4),
            6: (8, 4, 4, 1),
            248: (9, 8, 1, 1),
            251: (8, 6, 1, 16),
        }
        if kind not in layouts:
            raise ValueError(f"Unsupported RFB client message {kind}")
        header, offset, width, unit = layouts[kind]
        if len(self._buffer) < header:
            return None
        count = int.from_bytes(
            self._buffer[offset : offset + width], "big", signed=kind == 6
        )
        # Extended clipboard encodes its bounded payload with a negative S32.
        count = abs(count)
        length = header + count * unit
        if length > self.MAX_PAYLOAD:
            raise ValueError("RFB payload too large")
        if kind == 248 and count > 64:
            raise ValueError("Invalid RFB fence")
        return length
