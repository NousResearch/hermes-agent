from gateway.response_filters import (
    ends_with_partial_loop_complete_marker,
    strip_trailing_loop_complete_marker,
)


def test_loop_complete_display_filter_is_fence_aware():
    assert strip_trailing_loop_complete_marker("Done.\nLOOP_COMPLETE") == "Done."
    assert strip_trailing_loop_complete_marker("LOOP_COMPLETE") == ""
    assert strip_trailing_loop_complete_marker("Mention LOOP_COMPLETE in prose") == "Mention LOOP_COMPLETE in prose"
    fenced = "```text\nLOOP_COMPLETE\n```"
    assert strip_trailing_loop_complete_marker(fenced) == fenced


def test_loop_complete_partial_marker_only_matches_top_level_tail():
    assert ends_with_partial_loop_complete_marker("Done.\nLOOP_COM")
    assert ends_with_partial_loop_complete_marker("Done.\nLOOP_COMPLETE")
    assert not ends_with_partial_loop_complete_marker("```\nLOOP_COM")
    assert not ends_with_partial_loop_complete_marker("Mention LOOP_COM in prose")


def test_loop_complete_split_holds_the_whole_trailing_marker_run():
    """A repeated marker must be held as one run so none of it flashes while streaming."""
    from gateway.response_filters import split_trailing_loop_complete_marker

    assert split_trailing_loop_complete_marker("Done.\nLOOP_COMPLETE\nLOOP_COMPLETE") == (
        "Done.\n", "LOOP_COMPLETE\nLOOP_COMPLETE",
    )
    assert split_trailing_loop_complete_marker("Done.\nLOOP_COMPLETE\n\nLOOP_COM") == (
        "Done.\n", "LOOP_COMPLETE\n\nLOOP_COM",
    )
    assert split_trailing_loop_complete_marker("Done.\nLOOP_COM") == ("Done.\n", "LOOP_COM")
    assert split_trailing_loop_complete_marker("```\nLOOP_COMPLETE\n```\nLOOP_COM") == (
        "```\nLOOP_COMPLETE\n```\n", "LOOP_COM",
    )


def test_loop_complete_split_uses_released_fence_context():
    """A fence opened in an earlier chunk and closed in this one: the trailing marker after
    the closing fence is control text and must be held, not released."""
    from gateway.response_filters import split_trailing_loop_complete_marker

    seen = "Example:\n```text\nLOOP_COMPLETE\n"
    assert split_trailing_loop_complete_marker("```\nLOOP_COMPLETE", context=seen) == (
        "```\n", "LOOP_COMPLETE",
    )
    assert split_trailing_loop_complete_marker("```\nLOOP_COM", context=seen) == ("```\n", "LOOP_COM")
    # Still inside the open fence: nothing to hold.
    assert split_trailing_loop_complete_marker("more\nLOOP_COMPLETE", context=seen) == (
        "more\nLOOP_COMPLETE", "",
    )
    # A held run spanning released context only splits this chunk.
    assert split_trailing_loop_complete_marker("LOOP_COMPLETE", context="Done.\nLOOP_COMPLETE\n") == (
        "", "LOOP_COMPLETE",
    )
