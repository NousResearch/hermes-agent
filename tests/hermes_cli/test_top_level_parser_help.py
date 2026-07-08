from hermes_cli._parser import build_top_level_parser


def test_root_help_lists_webapp_surface_examples():
    help_text = build_top_level_parser()[0].format_help()

    assert "hermes dashboard              Start web UI dashboard" in help_text
    assert "hermes webapp                 Start browser-native workspace" in help_text
    assert "hermes webapp --stop          Stop running Hermes web server processes" in help_text
    assert "hermes webapp --status        List running Hermes web server processes" in help_text
