from agent.steer_media import (
    normalize_image_paths,
    persist_image_correction_text,
    supports_image_redirect,
    text_mode_image_correction,
)


def test_normalize_image_paths_drops_missing(tmp_path):
    real = tmp_path / "ok.png"
    real.write_bytes(b"x")
    assert normalize_image_paths([str(real), str(tmp_path / "gone.png"), "", str(real)]) == [
        str(real)
    ]


def test_persist_and_text_mode_include_caption(tmp_path):
    shot = tmp_path / "shot.png"
    shot.write_bytes(b"x")
    persist = persist_image_correction_text("see the error", [str(shot)])
    assert persist.startswith("see the error")
    assert "@image:" in persist
    text_mode = text_mode_image_correction("see the error", [str(shot)])
    assert "vision_analyze" in text_mode
    assert "see the error" in text_mode


def test_codex_does_not_support_image_redirect():
    class _A:
        api_mode = "codex_app_server"

    class _B:
        api_mode = "chat_completions"

    assert supports_image_redirect(_A()) is False
    assert supports_image_redirect(_B()) is True
