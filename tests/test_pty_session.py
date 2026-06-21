from hermes_cli.pty_session import RingBuffer


def test_ringbuffer_keeps_everything_under_capacity():
    rb = RingBuffer(10)
    rb.append(b"abc")
    rb.append(b"def")
    assert rb.snapshot() == b"abcdef"
    assert rb.truncated is False


def test_ringbuffer_drops_oldest_over_capacity():
    rb = RingBuffer(4)
    rb.append(b"abcdef")          # 6 bytes into a 4-byte buffer
    assert rb.snapshot() == b"cdef"
    assert rb.truncated is True


def test_ringbuffer_truncation_across_appends():
    rb = RingBuffer(3)
    rb.append(b"ab")
    rb.append(b"cd")             # now "abcd" -> keep "bcd"
    assert rb.snapshot() == b"bcd"
    assert rb.truncated is True
