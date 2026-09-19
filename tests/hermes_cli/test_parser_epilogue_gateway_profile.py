"""The ``--help`` epilogue must cover the profile-scoped gateway lifecycle.

``-p/--profile`` is consumed before argparse (``main._apply_profile_override``), so it never
appears among the parser's option rows — the epilogue examples are the only place ``--help``
can teach the ``hermes -p <profile> gateway <action>`` form that generated service units and
user-facing copy already rely on. Contract, not snapshot: assert the verbs and the profile flag
are named in the RENDERED help (the wiring, not just the constant), not the exact wording.
"""

from hermes_cli._parser import PRE_ARGPARSE_INHERITED_FLAGS, build_top_level_parser


def _rendered_help() -> str:
    return build_top_level_parser()[0].format_help()


def test_rendered_help_documents_the_profile_scoped_command_form():
    help_text = _rendered_help()
    assert "hermes -p <profile>" in help_text
    assert "--profile" in help_text
    assert "hermes -p coder gateway stop" in help_text
    # The flag stays pre-argparse: documented in the epilogue, never registered as an option.
    assert ("-p", True) in PRE_ARGPARSE_INHERITED_FLAGS
    assert "-p PROFILE" not in help_text and "--profile PROFILE" not in help_text


def test_rendered_help_documents_the_gateway_service_verbs():
    help_text = _rendered_help()
    for verb in ("install", "start", "stop", "status"):
        assert f"hermes gateway {verb}" in help_text, verb
