from argparse import Namespace

from hermes_cli.browser_qa import build_browser_qa_prompt, cmd_browser_qa


def test_build_browser_qa_prompt_contains_repeatable_requirements(tmp_path):
    output = tmp_path / "qa"
    prompt = build_browser_qa_prompt(
        url="http://127.0.0.1:3000",
        scope="login and checkout",
        output=str(output),
        max_pages=3,
        notes="Use demo user only.",
    )

    assert "Target URL: http://127.0.0.1:3000" in prompt
    assert "Scope: login and checkout" in prompt
    assert "Maximum pages/journeys to inspect: 3" in prompt
    assert f"Output directory: {output}" in prompt
    assert "browser_console" in prompt
    assert "failed requests" in prompt
    assert "not MEDIA: tags" in prompt
    assert "Use demo user only." in prompt


def test_cmd_browser_qa_preloads_dogfood_and_browser_toolsets():
    captured = {}

    def fake_chat_runner(args):
        captured["args"] = args
        return "ran"

    args = Namespace(
        url="https://example.test",
        scope=None,
        output="./qa-out",
        max_pages=5,
        notes=None,
        skills=["github-pr-workflow"],
        toolsets="terminal",
    )

    result = cmd_browser_qa(args, fake_chat_runner)

    assert result == "ran"
    prepared = captured["args"]
    assert "dogfood" in prepared.skills
    assert "github-pr-workflow" in prepared.skills
    assert prepared.toolsets == "terminal,browser,vision,file"
    assert prepared.source == "browser-qa"
    assert prepared.tui is False
    assert prepared.query.startswith("Run a browser QA mission")
    assert "Target URL: https://example.test" in prepared.query


def test_cmd_browser_qa_does_not_duplicate_skill_or_toolsets():
    captured = {}

    def fake_chat_runner(args):
        captured["args"] = args

    args = Namespace(
        url="https://example.test",
        skills=["dogfood"],
        toolsets="browser,file",
    )

    cmd_browser_qa(args, fake_chat_runner)

    prepared = captured["args"]
    assert prepared.skills == ["dogfood"]
    assert prepared.toolsets == "browser,file,vision"
