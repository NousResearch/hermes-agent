import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / ".github/actions/hermes-pr-docs/scripts/run.py"


def load_mod():
    spec = importlib.util.spec_from_file_location("pr_action_mod", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resolve_github_token_app_mode(monkeypatch):
    mod = load_mod()

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_APP_ID", "123")
    monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "456")
    monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY", "test-key")

    monkeypatch.setattr(mod, "get_github_app_token", lambda *args, **kwargs: "app-token")

    assert mod.resolve_github_token() == "app-token"


def test_emit_github_output_writes_github_output_file(tmp_path, monkeypatch):
    mod = load_mod()
    out_file = tmp_path / "github_output.txt"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out_file))

    mod.emit_github_output("comment_body", "hello world")

    assert out_file.read_text(encoding="utf-8") == "comment_body<<EOF\nhello world\nEOF\n"


def test_action_exports_review_mode_input():
    action_path = Path(__file__).resolve().parents[1] / ".github/actions/hermes-pr-docs/action.yml"
    text = action_path.read_text(encoding="utf-8")

    assert "review_mode:" in text
    assert "INPUT_REVIEW_MODE: ${{ inputs.review_mode }}" in text
