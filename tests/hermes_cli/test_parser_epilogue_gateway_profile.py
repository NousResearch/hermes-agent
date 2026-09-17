"""The ``--help`` epilogue must cover the profile-scoped gateway lifecycle.

``-p/--profile`` is consumed before argparse (``main._apply_profile_override``), so it never
appears among the parser's option rows — the epilogue examples are the only place ``--help``
can teach the ``hermes -p <profile> gateway <action>`` form that generated service units and
user-facing copy already rely on (#114495). Contract, not snapshot: assert the verbs and the
profile flag are named, not the exact wording.
"""

from hermes_cli._parser import _EPILOGUE


def test_epilogue_documents_the_profile_scoped_command_form():
    assert "-p <profile>" in _EPILOGUE
    assert "hermes -p coder gateway stop" in _EPILOGUE


def test_epilogue_documents_the_gateway_service_verbs():
    for verb in ("gateway start", "gateway stop", "gateway install"):
        assert f"hermes {verb}" in _EPILOGUE
