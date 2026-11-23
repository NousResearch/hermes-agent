"""A payload the copy protocol cannot rebuild must not cost the viewer its completion frame.

``SessionEvents.publish`` deep-copies the payload so a publisher that mutates it afterwards cannot
reach subscribers. ``copy``/``pickle`` rebuild a ``str`` subclass as ``cls.__new__(cls, value)``
via ``copyreg.__newobj__``, so a subclass needing a second ``__new__`` argument raised TypeError
there. ``publish`` had no guard: ``message.complete`` was never written, and the viewer kept
rendering the turn's last tool call as still running after the turn had settled.
"""
import logging

from gateway.session_events import SessionEvents


class _TwoArgStr(str):
    """The production shape: a ``str`` subclass whose ``__new__`` needs a second argument."""

    footer: str

    def __new__(cls, text, footer):
        instance = super().__new__(cls, f"{text}\n\n{footer}")
        instance.footer = footer
        return instance


def _published(stream):
    epoch, _ = stream.watermark()
    replay = stream.since(epoch, 0)
    assert replay['count'] == 1
    return replay['events'][0]['payload']


def test_an_uncopyable_payload_still_publishes_its_frame(caplog):
    stream = SessionEvents()
    with caplog.at_level(logging.WARNING, logger='gateway.session_events'):
        stream.publish('s', {'text': 'done', 'reply': _TwoArgStr('body', 'FOOTER')})

    published = _published(stream)
    assert published['text'] == 'done'
    assert published['reply'] == 'body\n\nFOOTER', 'the completion went out instead of being dropped'
    assert 'was not copyable' in caplog.text


def test_a_copyable_payload_is_still_isolated_from_later_mutation():
    stream = SessionEvents()
    payload = {'text': 'done', 'meta': {'tokens': 4}}
    stream.publish('s', payload)
    payload['meta']['tokens'] = 99

    assert _published(stream)['meta']['tokens'] == 4, 'the published frame aliased the caller payload'
