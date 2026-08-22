"""Cron media delivery: an unavailable gateway loop must not drop the batch.

``_send_media_via_adapter`` walks the per-file loop and reports one error
string per attachment (docstring contract: "Returns a list of per-file error
strings (empty when every attachment delivered)"). When the gateway loop is
unavailable (``safe_schedule_threadsafe`` returns ``None``), the loop used to
``return`` right after the FIRST file's error — every remaining attachment was
silently dropped: never attempted, never reported in the job's run status.
That is exactly the silent-drop class the manual-run media fix (22f0f2229)
claims to have eliminated.

This file pins the per-file contract for the unavailable-loop branch.
"""

import pytest

from cron.scheduler import _send_media_via_adapter


@pytest.fixture()
def unavailable_loop(monkeypatch):
    """Make every schedule attempt fail with an unavailable gateway loop.

    Returns a recorder list of every coroutine that WAS scheduled, so tests
    can assert that all files were attempted (not just the first one).
    """
    attempted = []

    def fake_schedule_threadsafe(coro, loop):
        attempted.append(coro)
        return None  # gateway loop unavailable

    monkeypatch.setattr(
        "agent.async_utils.safe_schedule_threadsafe", fake_schedule_threadsafe
    )
    # Keep path-policy validation out of the picture: these tests target the
    # scheduling loop, not the media-path filter.
    monkeypatch.setattr(
        "gateway.platforms.base.BasePlatformAdapter.filter_media_delivery_paths",
        staticmethod(lambda files: files),
    )
    monkeypatch.setattr(
        "gateway.platforms.base.validate_media_delivery_path",
        lambda path: path,
    )
    return attempted


class _StubAdapter:
    """Minimal adapter with every routing target the send loop may build."""

    platform = "stub"

    async def send_voice(self, **kw):  # pragma: no cover - never awaited
        pass

    async def send_image_file(self, **kw):  # pragma: no cover - never awaited
        pass

    async def send_video(self, **kw):  # pragma: no cover - never awaited
        pass

    async def send_document(self, **kw):  # pragma: no cover - never awaited
        pass


def _make_media_files(tmp_path, names):
    files = []
    for name in names:
        p = tmp_path / name
        p.write_bytes(b"x")
        files.append((str(p), False))
    return files


class TestUnavailableLoopReportsEveryFile:
    """Per-file error contract when the gateway loop is unavailable."""

    def test_multi_file_batch_reports_every_attachment(self, tmp_path, unavailable_loop):
        files = _make_media_files(tmp_path, ["a.mp3", "b.pdf", "c.png"])

        errors = _send_media_via_adapter(
            _StubAdapter(), "C1", files, None,
            loop=object(), job={"id": "job-multi"},
        )

        # Every file was attempted, none silently skipped.
        assert len(unavailable_loop) == len(files)
        # Every file produced its own error entry — the docstring's
        # "per-file error strings" contract, so the run status shows the
        # full drop instead of just the first file.
        assert len(errors) == len(files)
        for media_path, _v in files:
            assert any(
                media_path in err and "gateway loop unavailable" in err
                for err in errors
            )
        # An unavailable loop means the coroutines were never scheduled, so
        # nothing will ever await them. Each must be closed (cr_frame is None
        # after close(); a never-started coroutine keeps its frame) or Python
        # emits "coroutine was never awaited" per attachment and leaks the
        # frames.
        assert all(coro.cr_frame is None for coro in unavailable_loop)

    def test_single_file_still_reports_one_error(self, tmp_path, unavailable_loop):
        files = _make_media_files(tmp_path, ["clip.mp3"])

        errors = _send_media_via_adapter(
            _StubAdapter(), "C1", files, None,
            loop=object(), job={"id": "job-single"},
        )

        assert len(unavailable_loop) == 1
        assert len(errors) == 1
        assert "gateway loop unavailable" in errors[0]
        assert unavailable_loop[0].cr_frame is None

    def test_voice_and_document_routes_both_attempted(self, tmp_path, unavailable_loop):
        # Voice (a.mp3) and document (b.pdf) exercise the two main routing
        # branches; both must still be attempted when the loop is down.
        files = _make_media_files(tmp_path, ["a.mp3", "b.pdf"])

        _send_media_via_adapter(
            _StubAdapter(), "C1", files, None,
            loop=object(), job={"id": "job-route"},
        )

        assert len(unavailable_loop) == 2
        assert all(coro.cr_frame is None for coro in unavailable_loop)


class TestNoUnawaitedCoroutineWarnings:
    """The unavailable-loop path must not leak unawaited coroutines.

    Runs the multi-file scenario with RuntimeWarning elevated to an error;
    a leaked (unclosed) coroutine raises ``RuntimeWarning: coroutine ... was
    never awaited`` at teardown and fails the test.
    """

    @pytest.mark.filterwarnings("error::RuntimeWarning")
    def test_unavailable_loop_raises_no_runtime_warnings(
        self, tmp_path, unavailable_loop
    ):
        files = _make_media_files(tmp_path, ["a.mp3", "b.pdf", "c.png"])

        errors = _send_media_via_adapter(
            _StubAdapter(), "C1", files, None,
            loop=object(), job={"id": "job-warn"},
        )

        assert len(errors) == len(files)
        assert all(coro.cr_frame is None for coro in unavailable_loop)
