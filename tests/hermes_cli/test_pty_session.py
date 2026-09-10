"""Tests for hermes_cli/pty_session.py — PTY session registry."""


def test_ring_buffer_maxlen():
    from hermes_cli.pty_session import RingBuffer
    buf = RingBuffer(maxlen=3)
    buf.append(1)
    buf.append(2)
    buf.append(3)
    buf.append(4)
    assert len(buf) == 3
    assert list(buf) == [2, 3, 4]


def test_registry_full_is_exception():
    from hermes_cli.pty_session import RegistryFull
    exc = RegistryFull("full")
    assert isinstance(exc, Exception)
