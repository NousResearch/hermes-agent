"""Tests for opt-in comment descriptions in compact gateway tool progress."""

import copy
import unicodedata

import pytest


@pytest.mark.parametrize(
    ("tool_name", "args", "expected"),
    [
        (
            "terminal",
            {"command": "# Prüfe Branch und Status #\nprintf 'ok\\n'"},
            "Prüfe Branch und Status",
        ),
        (
            "execute_code",
            {"code": "\t# Analyze the test results\r\nprint('ok')"},
            "Analyze the test results",
        ),
        (
            "terminal",
            {"command": "# Safe label\rSECRET_OR_CODE"},
            "Safe label",
        ),
        ("terminal", {"command": "\r# Hidden after blank first line"}, None),
        ("terminal", {"command": "# Comment only"}, "Comment only"),
        ("terminal", {"command": "#tag\necho ok"}, "tag"),
        ("terminal", {"command": "##\necho ok"}, "#"),
        ("terminal", {"command": "## Heading ##\necho ok"}, "# Heading ##"),
        ("terminal", {"command": "# #\necho ok"}, None),
        ("terminal", {"command": "#   #\necho ok"}, None),
        (
            "terminal",
            {"command": "# Keep #hashtag and C#\necho ok"},
            "Keep #hashtag and C#",
        ),
        ("terminal", {"command": "# Keep Label#\necho ok"}, "Keep Label#"),
        ("terminal", {"command": "#! /usr/bin/env bash\necho ok"}, None),
        ("terminal", {"command": "\n# Hidden by blank first line\necho ok"}, None),
        ("terminal", {"command": "   \n# Hidden by whitespace first line"}, None),
        ("terminal", {"command": "#    \necho ok"}, None),
        ("terminal", {"command": "echo '# not a leading comment'"}, None),
        ("terminal", {"command": " \t# ASCII indentation\necho ok"}, "ASCII indentation"),
        ("terminal", {"command": "\u00a0# Not ASCII indentation\necho ok"}, None),
        ("terminal", {"command": "\v# Not ASCII indentation\necho ok"}, None),
        ("terminal", {"command": "\f# Not ASCII indentation\necho ok"}, None),
        ("terminal", {}, None),
        ("terminal", {"command": ""}, None),
        ("terminal", {"command": None}, None),
        ("terminal", {"command": 42}, None),
        ("terminal", None, None),
        ("browser_exec", {"code": "# Browser label\nreturn 1"}, None),
    ],
)
def test_extract_tool_progress_comment_description(tool_name, args, expected):
    from agent.display import extract_tool_progress_comment_description

    assert extract_tool_progress_comment_description(tool_name, args) == expected


def test_prepare_tool_progress_comment_description_truncates_visible_text():
    from agent.display import prepare_tool_progress_comment_description

    preview = prepare_tool_progress_comment_description(
        "terminal",
        {"command": "# " + "x" * 80 + "\necho ok"},
        max_len=20,
    )

    assert preview is not None
    assert preview.text == "x" * 17 + "．．．"
    assert preview.truncated is True
    assert preview.url is None


def test_prepare_tool_progress_comment_description_neutralizes_markup_and_mentions():
    from agent.display import prepare_tool_progress_comment_description

    preview = prepare_tool_progress_comment_description(
        "execute_code",
        {
            "code": (
                "# **bold** [link](https://example.test) @channel "
                "<@123> <!channel> `code` plain.ext a=b "
                "http://localhost/path //localhost/path "
                ":party: &lt;@everyone&gt; @\u200beveryone "
                "\u202e\u0000\nprint('ok')"
            )
        },
        max_len=0,
    )

    assert preview is not None
    assert not set("\\`*_{}[]()<>#+-=.!|~@&:/").intersection(preview.text)
    for active in (
        "**bold**",
        "[link](https://example.test)",
        "@channel",
        "<@123>",
        "<!channel>",
        "`code`",
        "plain.ext",
        "a=b",
        "http://localhost/path",
        "//localhost/path",
        ":party:",
        "&lt;@everyone&gt;",
        "@everyone",
        "\u202e",
        "\u0000",
    ):
        assert active not in preview.text
    compatibility_text = unicodedata.normalize("NFKC", preview.text)
    assert "plain.ext" in compatibility_text
    assert "a=b" in compatibility_text
    assert preview.url is None


def test_prepare_force_redacts_recognizable_secrets_before_rendering():
    from agent.display import prepare_tool_progress_comment_description

    secret = "sk-proj-" + "L" * 24
    preview = prepare_tool_progress_comment_description(
        "execute_code",
        {"code": f"# Inspect {secret}\nprint('ok')"},
        max_len=0,
    )

    assert preview is not None
    assert secret not in unicodedata.normalize("NFKC", preview.text)


def test_prepare_force_redacts_url_credentials_before_cap_without_mutating_args():
    from agent.display import prepare_tool_progress_comment_description

    args = {
        "code": (
            "# Fetch https://alice:wolfpass9@example.test/path"
            "?access_token=opaquevalue123&mode=safe\nprint('ok')"
        )
    }
    original_args = copy.deepcopy(args)

    full_preview = prepare_tool_progress_comment_description(
        "execute_code",
        args,
        max_len=0,
    )
    capped_preview = prepare_tool_progress_comment_description(
        "execute_code",
        args,
        max_len=27,
    )

    assert full_preview is not None
    assert capped_preview is not None
    assert "wolfpass9" not in full_preview.text
    assert "opaquevalue123" not in full_preview.text
    assert "wolf" not in capped_preview.text
    assert capped_preview.truncated is True
    assert args == original_args


def test_prepare_normalizes_invisible_token_splits_before_redaction():
    from agent.display import prepare_tool_progress_comment_description

    obfuscated_secret = "sk-\u200bproj-" + "Q" * 24
    preview = prepare_tool_progress_comment_description(
        "execute_code",
        {"code": f"# Inspect {obfuscated_secret}\nprint('ok')"},
        max_len=0,
    )

    assert preview is not None
    assert "Q" * 24 not in preview.text


@pytest.mark.parametrize("invisible", ["\u034f", "\ufe0f", "\u115f"])
def test_prepare_removes_non_control_default_ignorables_before_redaction(invisible):
    from agent.display import prepare_tool_progress_comment_description

    obfuscated_secret = "sk-" + invisible + "proj-" + "R" * 24
    preview = prepare_tool_progress_comment_description(
        "execute_code",
        {"code": f"# Inspect {obfuscated_secret}\nprint('ok')"},
        max_len=0,
    )

    assert preview is not None
    assert invisible not in preview.text
    assert "R" * 24 not in preview.text


@pytest.mark.parametrize("invisible", ["\u034f", "\ufe0f", "\u115f"])
def test_prepare_returns_none_for_non_control_default_ignorable_only_label(invisible):
    from agent.display import prepare_tool_progress_comment_description

    assert (
        prepare_tool_progress_comment_description(
            "terminal",
            {"command": f"# {invisible}\necho ok"},
            max_len=40,
        )
        is None
    )


def test_prepare_forces_all_secret_classes_when_redaction_is_globally_disabled(
    monkeypatch,
):
    import agent.redact as redact
    from agent.display import prepare_tool_progress_comment_description

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    prefix_secret = "sk-proj-" + "S" * 24
    code = (
        f"# Inspect {prefix_secret} "
        "https://alice:wolfpass9@example.test/path"
        "?access_token=opaquevalue123\nprint('ok')"
    )

    preview = prepare_tool_progress_comment_description(
        "execute_code",
        {"code": code},
        max_len=0,
    )

    assert preview is not None
    compatibility_text = unicodedata.normalize("NFKC", preview.text)
    for secret in (prefix_secret, "wolfpass9", "opaquevalue123"):
        assert secret not in compatibility_text


def test_tool_verb_can_use_explicit_turn_setting_instead_of_module_global():
    from agent.display import get_tool_verb, set_friendly_tool_labels

    set_friendly_tool_labels(False)
    try:
        assert get_tool_verb("terminal", friendly_labels=True) == "Running"
        assert get_tool_verb("terminal", friendly_labels=False) is None
        assert get_tool_verb("terminal") is None
        set_friendly_tool_labels(True)
        assert get_tool_verb("terminal") == "Running"
    finally:
        set_friendly_tool_labels(True)


def test_prepare_returns_none_when_only_control_formatting_survives_parser():
    from agent.display import prepare_tool_progress_comment_description

    assert (
        prepare_tool_progress_comment_description(
            "terminal",
            {"command": "# \u202e\u200b\necho ok"},
            max_len=40,
        )
        is None
    )


def test_prepare_returns_none_when_over_cap_input_is_only_control_formatting():
    from agent.display import prepare_tool_progress_comment_description

    assert (
        prepare_tool_progress_comment_description(
            "terminal",
            {"command": "# " + "\u202e" * 80 + "\necho ok"},
            max_len=20,
        )
        is None
    )
