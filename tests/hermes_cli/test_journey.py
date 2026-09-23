def test_open_in_editor_supports_quoted_executable_path(tmp_path, monkeypatch):
    from hermes_cli.journey import _open_in_editor

    editor = tmp_path / "fake editor"
    editor.write_text(
        "#!/bin/sh\n"
        "printf 'edited\\n' > \"$1\"\n",
        encoding="utf-8",
    )
    editor.chmod(0o755)

    monkeypatch.setenv("EDITOR", f'"{editor}"')
    monkeypatch.delenv("VISUAL", raising=False)

    assert _open_in_editor("original\n", suffix=".txt") == "edited\n"
