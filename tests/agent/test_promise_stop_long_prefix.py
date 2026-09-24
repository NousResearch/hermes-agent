"""A diagnostic preamble must not hide a dangling action at the end."""
import pytest
from agent.promise_stop import looks_like_immediate_action_promise

@pytest.mark.parametrize('tail', [
    'Fixing all of it now.',
    "Fixing both defects and making the tests permanent (`/tmp` is being wiped between calls, which is why the regression file vanished):",
    "Moving them into the profile where they'll persist, fixing the two real defects the run exposed, and installing the skill rules — all as writes now, then one verification run:",
])
def test_diagnostic_preamble_does_not_hide_action_promise(tail):
    preamble = 'The previous check exposed an error in timestamp parsing. ' * 10
    assert looks_like_immediate_action_promise(preamble + tail)
