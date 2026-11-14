"""The cron delivery mirror label attributes the brief and closes it (#118863).

The mirror is a USER-role row (alternation, see ``_maybe_mirror_cron_delivery``) and mirror
metadata is dropped at the SQLite boundary, so the text label is the model's only cue about
where the brief came from. When the user replies, ``_merge_consecutive_users`` folds the reply
into the mirror row with a bare ``"\\n\\n"``. The label therefore has to carry two facts:

- attribution: the brief is the agent's own already-sent message, not something the user (or
  an operator) wrote;
- boundary: the brief is closed, so the merged turn still shows where the delivery ends and
  the reply begins.
"""

from agent.agent_runtime_helpers import _merge_consecutive_users
from cron.scheduler_delivery import _cron_mirror_message


def test_mirror_label_attributes_and_closes_before_merged_user_reply():
    mirror = _cron_mirror_message({"id": "j1", "name": "Morning Brief"}, "Market movers today")
    first_line = mirror.split("\n", 1)[0]
    assert first_line.startswith("[Cron delivery: Morning Brief]")
    assert "your own scheduled message" in first_line
    assert mirror.endswith("[Cron delivery end: Morning Brief]")

    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": mirror},
        {"role": "user", "content": "what about tech stocks?"},
    ])
    assert repairs == 1 and len(merged) == 1
    content = merged[0]["content"]
    end = content.index("[Cron delivery end: Morning Brief]")
    assert content.index("Market movers today") < end < content.index("what about tech stocks?")

def test_mirror_label_keeps_multiline_job_name_inside_delimiters():
    mirror = _cron_mirror_message(
        {"id": "j1", "name": "Morning\nBrief"}, "Market movers today"
    )
    assert mirror.splitlines() == [
        "[Cron delivery: Morning Brief] (your own scheduled message, already sent to the user)",
        "Market movers today",
        "[Cron delivery end: Morning Brief]",
    ]

    merged, repairs = _merge_consecutive_users([
        {"role": "user", "content": mirror},
        {"role": "user", "content": "what about tech stocks?"},
    ])
    assert repairs == 1
    assert merged[0]["content"].endswith(
        "[Cron delivery end: Morning Brief]\n\nwhat about tech stocks?"
    )


def test_mirror_label_collapses_line_breaks_but_keeps_other_name_whitespace():
    # U+3000 is part of the stored name that resolve_job_ref matches on; only line breaks go.
    mirror = _cron_mirror_message(
        {"id": "j1", "name": "\n朝の\u3000ブリーフ\n\n   Daily"}, "Market movers today"
    )
    opener, _brief, closer = mirror.splitlines()
    assert opener.startswith("[Cron delivery: 朝の\u3000ブリーフ Daily]")
    assert closer == "[Cron delivery end: 朝の\u3000ブリーフ Daily]"
