"""Text in fenced or inline code is quoted, never a delivery directive.

Markdown renders code literally, so an ``![alt](url)`` image tag, a ``MEDIA:`` tag or a bare
local path inside a code span is something the agent is showing (a markup example, a quoted
client message), not an attachment it asked for. ``extract_local_files`` and ``extract_media``
already skipped code; ``extract_images`` did not, so a quoted image tag was sent as a photo.

Fences follow CommonMark: a fence closes only at a line holding nothing but a backtick run at
least as long as the opening one, so a ``` in the middle of a quoted line, or a shorter run
inside a longer fence, does not end the block and expose the text after it. A fence left open
runs to the end of the reply, as CommonMark renders it.
"""

from gateway.platforms.base import BasePlatformAdapter

IMG = "https://example.invalid/a.png"


# ---------------------------------------------------------------------------
# extract_images
# ---------------------------------------------------------------------------


def test_image_tag_quoted_in_a_fence_stays_text():
    reply = f"Клиент пишет:\n```text\nСкрин ![x]({IMG})\n```\nРазберусь."

    images, cleaned = BasePlatformAdapter.extract_images(reply)

    assert images == []
    assert cleaned == reply


def test_image_tag_in_inline_code_stays_text():
    reply = f"Разметка картинки: `![x]({IMG})`"

    images, cleaned = BasePlatformAdapter.extract_images(reply)

    assert images == []
    assert cleaned == reply


def test_html_img_in_code_stays_text():
    reply = f'Пример:\n```html\n<img src="{IMG}">\n```'

    images, cleaned = BasePlatformAdapter.extract_images(reply)

    assert images == []
    assert cleaned == reply


def test_image_outside_code_is_delivered_and_its_quoted_twin_kept():
    """Only the tag outside code is removed, even when the quoted one has the same URL."""
    reply = f"![chart]({IMG})\n\nРазметка: `![chart]({IMG})`"

    images, cleaned = BasePlatformAdapter.extract_images(reply)

    assert images == [(IMG, "chart")]
    assert cleaned == f"Разметка: `![chart]({IMG})`"


def test_image_after_a_closed_fence_is_delivered():
    reply = f"```\nprint(1)\n```\n![plot]({IMG})"

    images, cleaned = BasePlatformAdapter.extract_images(reply)

    assert images == [(IMG, "plot")]
    assert cleaned == "```\nprint(1)\n```"


def test_image_in_an_unclosed_fence_stays_text():
    reply = f"Лог:\n```text\nСкрин ![x]({IMG})"

    images, cleaned = BasePlatformAdapter.extract_images(reply)

    assert images == []
    assert cleaned == reply


def test_backticks_mid_line_do_not_close_the_fence_for_images():
    reply = f"```text\nне работает ``` ![x]({IMG})\n```"

    images, _ = BasePlatformAdapter.extract_images(reply)

    assert images == []


# ---------------------------------------------------------------------------
# Fence closing (shared by extract_local_files and extract_media)
# ---------------------------------------------------------------------------


def _real_file(tmp_path):
    path = tmp_path / "report.pdf"
    path.write_bytes(b"%PDF-1.4")
    return path.as_posix()


def test_backticks_mid_line_do_not_close_the_fence_for_local_files(tmp_path):
    path = _real_file(tmp_path)
    reply = f"Клиент пишет:\n```text\nНе работает принтер ``` {path}\n```"

    files, cleaned = BasePlatformAdapter.extract_local_files(reply)

    assert files == []
    assert cleaned == reply


def test_shorter_run_inside_a_longer_fence_does_not_close_it(tmp_path):
    path = _real_file(tmp_path)
    reply = f"````text\n```\n{path}\n````"

    files, _ = BasePlatformAdapter.extract_local_files(reply)

    assert files == []


def test_local_file_after_a_closed_fence_is_delivered(tmp_path):
    path = _real_file(tmp_path)
    reply = f"```text\nцитата\n```\nОтчёт: {path}"

    files, _ = BasePlatformAdapter.extract_local_files(reply)

    assert len(files) == 1


def test_local_file_in_an_unclosed_fence_is_not_delivered(tmp_path):
    path = _real_file(tmp_path)
    reply = f"```text\nцитата {path}"

    files, _ = BasePlatformAdapter.extract_local_files(reply)

    assert files == []


def test_backticks_mid_line_do_not_close_the_fence_for_media_tags():
    reply = "```text\nцитата ``` MEDIA:/tmp/quoted/report.pdf\n```"

    media, cleaned = BasePlatformAdapter.extract_media(reply)

    assert media == []
    assert cleaned == reply
