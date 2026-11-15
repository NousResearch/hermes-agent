"""A prefix match that a skipped or shorter match overlaps must still mask (#39654).

Both the control-split pass and the contiguous pass used to keep one match per position, so a
match that started inside another (``my-sk-learn-config.1//0…`` skipped as a glued identifier, or
``1//0…log-sk-abc.<key>`` ending at the ``.``) hid the key it ran on into.
"""

import time

import pytest

from agent.redact import redact_sensitive_text

G1 = "1//" + "0aS42YiA\u200b3ML0Entm5qgXXDNJHdAq"  # zero-width-split refresh token
GLC = "gl" + "c_PUxeylAbudU5oacBSIskLkgtDcQ"


@pytest.fixture(autouse=True)
def _ensure_redaction_enabled(monkeypatch):
    monkeypatch.delenv("HERMES_REDACT_SECRETS", raising=False)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", True)


def _assert_masked(text: str, *fragments: str) -> None:
    for out in (redact_sensitive_text(text, force=True),
                redact_sensitive_text(text, force=True, file_read=True, secret_file=True)):
        for fragment in fragments:
            assert fragment not in out, out


@pytest.mark.parametrize("sep", ["\n", "\t", "\r\n"])
@pytest.mark.parametrize("glued", ["my-sk-learn-config.", "log-sk-abc.", "my-sk-a."])
def test_key_after_a_skipped_glued_field_on_the_previous_line_is_masked(sep, glued):
    # The skipped first candidate ends in the identifier's tail; the gap-fill candidate that
    # started there crossed the stripped break and took the next line's ``sk-`` with it.
    _assert_masked(f"my-sk-learn-config.{G1}{sep}{glued}{GLC}\rx5\n", "3ML0Entm5qgX", "AbudU5oacBSI")


def test_key_after_a_glued_field_mid_line_is_masked():
    text = "x-sk-\tAKIAVji9tEIWRoJv\rxreYspK8jfM1oKkDiyAATZxx2SuQm" + f"my-sk-learn-config.{G1}\nmy-sk-learn-config.{GLC}\rx5\n"
    _assert_masked(text, "AbudU5oacBSI")


@pytest.mark.parametrize("split", ["", "\u200b"])
def test_key_inside_the_tail_of_a_refresh_token_match_is_masked(split):
    # The ``1//0`` match ends at the ``.``; the ``sk-`` match inside it runs on through the key.
    refresh = "1//" + "0zHbvuQAXdRVU" + split + "aPFrEcBpQOSnacj0hfv"
    _assert_masked(f"x {refresh}log-sk-abc.gl" + "c_dwWJmRyYkVqZMvgdmIT53yB end", "dwWJmRyYkVqZ")


@pytest.mark.parametrize("refresh", [G1, "1//" + "0zHbvuQAXdRVUaPFrEcBpQOSnacj0hfv"])
def test_key_past_a_match_contained_in_an_earlier_one_is_masked(refresh):
    # The ``hf_`` match ends inside the ``sk-`` match; the refresh token starts inside it too.
    sk = "sk-" + "proj-Ab3dEf6hIj9kLm2nOp5qRs8tUv-" + "hf_" + "AbCdEfGhIjKlMnOpQrStUvWxYz123456-xy_"
    _assert_masked(f"x {sk}{refresh} end", refresh[12:24])


def test_many_short_glued_lines_stay_linear():
    text = "my-sk-a.\n" * 11_112  # 100k chars, every line a skipped glued candidate
    t0 = time.perf_counter()
    assert redact_sensitive_text(text, force=True) == text
    assert time.perf_counter() - t0 < 2.0
