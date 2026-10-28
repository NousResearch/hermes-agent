"""The session-authority turn stamps its reply frames with the same wire identity as tui_gateway's ``_emit``.

A viewer upserts a re-sent delta or seal by ``turn_id``/``segment``/``offset`` instead of painting a second copy.
"""
from gateway.run_turn_runner import TurnRunner


class _Authority:
    def __init__(self):
        self.frames = []

    def publish_execution(self, session_id, generation, event_type, payload):
        self.frames.append((generation, event_type, payload))
        return True


def test_reply_frames_carry_one_cursor_per_execution_generation():
    authority = _Authority()
    runner = TurnRunner.__new__(TurnRunner)
    runner._approval_owner = (authority, 's', 1)
    for event_type, payload in (('message.delta', {'text': 'a\U0001F440'}), ('message.delta', {'text': 'b'}),
                                ('message.interim', {'text': 'a\U0001F440b'}), ('message.delta', {'text': 'c'}),
                                ('tool.start', {'name': 'terminal'})):
        runner._publish_execution(event_type, payload)
    runner._approval_owner = (authority, 's', 2)
    runner._publish_execution('message.delta', {'text': 'next'})

    stamps = [{k: p.get(k) for k in ('turn_id', 'segment', 'offset') if k in p} for _, _, p in authority.frames]
    first, second = stamps[0]['turn_id'], stamps[-1]['turn_id']
    assert stamps[:5] == [
        {'turn_id': first, 'segment': 0, 'offset': 0},
        {'turn_id': first, 'segment': 0, 'offset': 3},  # the emoji is two UTF-16 code units
        {'turn_id': first, 'segment': 0},
        {'turn_id': first, 'segment': 1, 'offset': 0},
        {},  # not a reply frame: unchanged
    ]
    assert stamps[5] == {'turn_id': second, 'segment': 0, 'offset': 0} and second != first
