"""An adviser row keeps one label per note, so the CLI can paint each note by severity."""

from hermes_cli.cli_agent_setup_mixin import _adviser_items


def test_adviser_items_keep_per_note_labels():
    content = "\n".join(
        [
            "[adviser — advisory only]",
            "- [concern · reviewer] read the import first",
            "- [blocker · reviewer] the test never ran",
            "[end adviser]",
        ]
    )

    assert _adviser_items(content) == [
        ("concern · reviewer", "read the import first"),
        ("blocker · reviewer", "the test never ran"),
    ]


def test_adviser_items_skip_the_bracketed_chrome():
    content = "[adviser — advisory only]\nplain advice from an older host\n[end adviser]"

    assert _adviser_items(content) == []
