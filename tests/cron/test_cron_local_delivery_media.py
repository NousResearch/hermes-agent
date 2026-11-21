"""A cron result delivered into a local (Desktop/Webapp) chat keeps its MEDIA attachments (#148449).

The local lane admits the delivery as a note in the chat's history. It used to receive the
tag-stripped text only, so every ``MEDIA:`` file a job produced vanished from the chat while the
run reported a clean delivery. The note now carries one ``MEDIA:`` line per file that passed the
media path policy (the filter every other lane applies) and a card naming the job, which the
Desktop paints as a collapsible result row whose report renders the media.
"""
import pytest

from cron.scheduler_delivery import _deliver_result


def _local_job(name="News radar"):
    return {"id": "job-local-media", "name": name, "deliver": "origin",
            "origin": {"platform": "local", "chat_id": "local-abc"}}


def _capture(monkeypatch):
    from gateway import session_cron
    seen = []

    def fake_deliver(home, chat_id, text, *, diagnostic=False, title=None):
        seen.append({"chat_id": chat_id, "text": text, "diagnostic": diagnostic, "title": title})

    monkeypatch.setattr(session_cron, "deliver_to_local_chat", fake_deliver)
    return seen


def test_local_note_keeps_each_deliverable_attachment_as_a_media_line(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    folder = tmp_path / "News Radar"  # a space, as in the Google Drive path of the field report
    folder.mkdir()
    gif = folder / "animation.gif"
    gif.write_bytes(b"GIF89a")
    page = folder / "index.html"
    page.write_text("<html></html>")
    missing = folder / "gone.png"
    seen = _capture(monkeypatch)

    error = _deliver_result(
        _local_job(), f"3 takes are ready.\nIndex: MEDIA:{page}\n\nMEDIA:{gif}\nMEDIA:{missing}",
        adapters=None, loop=None)

    assert [s["chat_id"] for s in seen] == ["local-abc"]
    lines = seen[0]["text"].splitlines()
    assert "3 takes are ready." in lines
    assert f"MEDIA:{page.resolve()}" in lines and f"MEDIA:{gif.resolve()}" in lines
    assert str(missing) not in seen[0]["text"]
    # The missing file is a policy drop, recorded as on every lane; the two delivered files are not.
    assert error is not None and "1 media attachment(s) dropped" in error, error


def test_text_only_local_note_gains_no_media_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    seen = _capture(monkeypatch)

    assert _deliver_result(_local_job(), "Nothing new today.", adapters=None, loop=None) is None
    assert "MEDIA:" not in seen[0]["text"]


@pytest.mark.parametrize("for_failure, outcome", [(False, "finished"), (True, "failed")])
def test_local_note_names_the_job_in_its_card(tmp_path, monkeypatch, for_failure, outcome):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    seen = _capture(monkeypatch)

    _deliver_result(_local_job("News radar: daily quick takes"), "report", adapters=None, loop=None,
                    for_failure=for_failure)

    assert seen[0]["title"] == f"Cron job 'News radar: daily quick takes' {outcome}"
    assert seen[0]["diagnostic"] is for_failure
